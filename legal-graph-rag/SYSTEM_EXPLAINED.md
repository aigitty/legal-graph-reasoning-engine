# Legal Graph RAG — Complete System Walkthrough

A deep, end-to-end explanation of how this LangGraph + Neo4j legal reasoning
engine works, written for someone new to LangGraph and with only basic Cypher.
Every claim is anchored in the actual code in this repo.

> **Companion file:** see `ARCHITECTURE_DIAGRAM.md` for the diagrams.

---

## Table of contents

0. [The two ways this project "runs"](#part-0)
1. [LangGraph in plain language](#part-1)
2. [The literal startup sequence](#part-2)
3. [How the graph is wired (`build_graph()`)](#part-3)
4. [The worked example, node by node](#part-4)
5. [The state object (the clipboard)](#part-5)
6. [Neo4j retrieval, with Cypher explained](#part-6)
7. [The tunable parameters](#part-7)
8. [Tracing & observability](#part-8)
9. [The architecture, top to bottom](#part-9)

---

<a name="part-0"></a>
## Part 0 — The two ways this project "runs"

There are **two separate entry points**, and conflating them is the #1 thing
that will trip you up.

| Entry point | What it is | LLM? | Agents? |
|---|---|---|---|
| `python main.py` | A **legacy terminal tester** for graph traversal only. Pure Neo4j. | No | No |
| `python graph_agent.py` | The **real agentic pipeline** — the LangGraph workflow. | Yes (3 calls) | Yes (the state machine) |

`main.py` literally prints *"No LLM. No agents. No guardrails."* (line 182). It
is a debugging harness to eyeball what traversal returns. Everything that matters
for the system is in **`graph_agent.py`**.

> **Known issue (honesty note):** in `ingest/graph_builder.py`, `main()`
> references `driver` on line 283 (the "Clearing existing graph data" block)
> **before** `driver` is assigned on line 298 — that is an `UnboundLocalError`
> as the code stands. The graph is already populated, so this ran in an earlier
> form; the clear-block was likely added later above the driver init. It is
> ingestion code, not part of the runtime, but worth knowing before anyone runs
> it live.

---

<a name="part-1"></a>
## Part 1 — LangGraph in plain language (the mental model)

Four concepts:

1. **State** — one shared data object that travels through the whole pipeline.
   Think of it as a **clipboard passed hand to hand**. Each worker reads what is
   on it and writes new things onto it. Here the clipboard is `LegalQueryState`
   (in `agents/graph_state.py`).

2. **Node** — a worker. Literally a Python function: it receives the clipboard
   (state), does one job, and returns a **small dict of just the fields it
   changed**. LangGraph merges that dict back into the master state. Nodes never
   mutate state in place; they return partial updates.

3. **Edge** — an arrow saying "after this node, go to that node." A plain edge is
   unconditional. A **conditional edge** runs a small **router function** that
   looks at the state and *returns a label*; a lookup table maps that label to
   the next node. That is how branching and looping happen.

4. **`START` / `END`** — two special sentinels marking where the graph begins and
   stops. `compile()` freezes the wiring into a runnable object; `invoke()`
   actually runs one query through it.

Keep "clipboard passed hand to hand" in your head — the rest is detail.

---

<a name="part-2"></a>
## Part 2 — The literal startup sequence (what runs first, second, third)

When you run `python graph_agent.py`:

**1. Module import side-effects (top of `graph_agent.py`):**
- `load_dotenv()` runs immediately at import. It reads `.env` into environment
  variables. It is placed *before* the node imports on purpose — the nodes build
  their LLM clients at import time and need `GOOGLE_CLOUD_PROJECT` etc. present.
- The `from agents.nodes... import ...` lines run. **Importing those modules does
  work**, because several build objects at module load:
  - `extraction_node.py` -> `_EXTRACTION_LLM = get_llm(...).with_structured_output(ExtractionResult)`
  - `sufficiency_node.py` -> `_SUFFICIENCY_LLM = get_llm(...).with_structured_output(SufficiencyVerdict)`
  - `synthesis_node.py` -> `_SYNTHESIS_LLM = get_llm(...)`
  - Each `get_llm()` (in `agents/llm.py`) reads `config.cfg` and builds a
    `ChatVertexAI` client. So **three Gemini clients are created before your
    query exists** — a deliberate "build once, reuse" optimization.
  - Importing `config` runs `cfg = Settings()` (bottom of `config.py`), loading
    and validating all settings from `.env` + defaults.
  - Importing anything touching `graph/db_connection.py` runs its body, which
    **opens the Neo4j driver** (a singleton) and raises immediately if creds are
    missing.

**2. `GRAPH = build_graph()` (module level):** constructs and **compiles** the
state machine. Happens once.

**3. The `if __name__ == "__main__":` block (bottom):** runs only when the file
is executed directly. It calls `run(...)` three times with hardcoded queries and
prints each via `_print_result`.

**4. `run(raw_query)`:** processes one query:
```python
out = GRAPH.invoke(
    {"raw_query": raw_query},
    config={"recursion_limit": cfg.LANGGRAPH_RECURSION_LIMIT},
)
```
`GRAPH.invoke({"raw_query": ...})` is where the pipeline starts. You hand it a
dict with just the raw query; LangGraph wraps it into a fresh `LegalQueryState`
(all other fields take defaults) and pushes it from `START` into the first node.

**Sequence:** load env -> build LLM clients + open Neo4j -> compile graph ->
invoke graph per query.

---

<a name="part-3"></a>
## Part 3 — How the graph is wired (`build_graph()`)

This function is the blueprint.

```python
builder = StateGraph(LegalQueryState)   # "the clipboard's shape is LegalQueryState"
```
This tells LangGraph the shared state is your pydantic `LegalQueryState`. That is
why every node can type its argument as `LegalQueryState` and trust the fields.

**Register the workers** (`add_node`):
```
entity_extraction, concept_grounding, graph_traversal, sufficiency_evaluator,
graph_expansion, answer_synthesis, output_guardrail, final_response
```

**Wire the arrows** (`add_edge` = unconditional):
```
START -> entity_extraction -> concept_grounding -> graph_traversal -> sufficiency_evaluator
graph_expansion -> graph_traversal          (the loop's "back" arrow)
answer_synthesis -> output_guardrail -> final_response -> END
```

**The one branch point** (`add_conditional_edges`):
```python
builder.add_conditional_edges(
    "sufficiency_evaluator",
    _route_after_sufficiency,                       # the router function
    {"expand": "graph_expansion", "end": "answer_synthesis"},
)
```
After `sufficiency_evaluator` runs, `_route_after_sufficiency(state)` returns
`"expand"` or `"end"`, and the dict translates that into the next node. **This is
the entire decision-making structure of the agent** — one fork.

`_route_after_sufficiency` logic:
- No verdict, or verdict says sufficient -> `"end"` (go synthesize).
- Already hit `MAX_ITERATIONS` retries -> `"end"`.
- Retrieval empty (nothing to expand from) -> `"end"`.
- Otherwise -> `"expand"` (loop back for a deeper retrieval).

**The LLM does not decide whether to loop.** The router is plain Python reading
counters. The LLM only *fills in* the sufficiency verdict; deterministic code
decides what to do with it (CLAUDE.md rule 4).

`builder.compile()` returns the runnable `GRAPH`. The `recursion_limit` (25) is
passed at `invoke()` time as a backstop so a wiring bug cannot loop forever.

---

<a name="part-4"></a>
## Part 4 — The worked example, node by node

Worked query:

> **"fired without notice after 3 years at a private company in Karnataka"**

The clipboard starts as `raw_query="fired without notice…"`, everything else
default.

### Node 1 — `entity_extraction_node` (`extraction_node.py`) — LLM call #1
- **Job:** turn messy English into structured fields.
- **Reads:** `state.raw_query`.
- **Internally:** if blank, returns `{"error": ...}`. Otherwise calls
  `_EXTRACTION_LLM.invoke([("system", SYSTEM_PROMPT), ("human", raw_query)])`. The
  `.with_structured_output(ExtractionResult)` wrapper forces Gemini to return JSON
  matching the `ExtractionResult` schema (`schemas.py`) — a typed object, not free
  text. On error it returns a graceful `{"error": ...}`.
- **Writes:** `extraction` = `ExtractionResult` with `legal_concepts`,
  `jurisdiction`, `employment_type`, `years_of_service`, `triggering_event`,
  `in_domain`, `safety_flag`.
- **Our run:** `legal_concepts=['wrongful termination', 'notice period']`,
  `jurisdiction='Karnataka'`, `in_domain=True`.
- **Next:** always -> `concept_grounding`.

> **LangGraph teaching point:** the node returned only `{"extraction": <obj>}`.
> LangGraph merged that one key into the clipboard; everything else is untouched.
> That is the partial-update model.

### Node 2 — `concept_grounding_node` (`grounding_node.py`) — deterministic
- **Job:** map the LLM's free-text concepts onto the **fixed 25-concept
  vocabulary** so traversal has a guaranteed entry point into the graph. No LLM.
- **Reads:** `state.extraction.legal_concepts` (falls back to `[state.raw_query]`).
- **Internally:** for each candidate calls `ground_query()` in `agents/ontology.py`
  (substring match against concept names/aliases, then fuzzy fallback). Unioned,
  dedup'd.
- **Writes:** `grounded_concepts`. If nothing grounds, writes a `warnings` entry.
- **Our run:** `['wrongful termination', 'notice period']`.
- **Next:** always -> `graph_traversal`.

### Node 3 — `graph_traversal_node` (`retrieval_node.py`) — deterministic, hits Neo4j
- **Job:** pull the actual statutory sections out of Neo4j for the grounded
  concepts. **This is the heart of Graph RAG** — retrieval is a graph walk, not an
  LLM guess.
- **Reads:** `grounded_concepts`, `max_hops` (starts at 2).
- **Internally:** for each concept calls `traverse(concept_name=...,
  max_hops=state.max_hops, max_sections=MAX_SECTIONS, exact=True)` from
  `graph/traversal.py`. Merges across concepts: dedup by `section_id`, keep
  strongest relevance ("primary" > "supporting"), union acts + CITES edges.
  `relevance_for()` tags primary vs supporting.
- **Writes:** `retrieval` = `RetrievalResult` with `.sections`
  (list of `SectionContext`), `.total_found`, `.confidence`, `.is_empty`, and the
  `.section_ids` property used later.
- **Our run:** `total_found=12` across IDA 1947 and KSEA 1961; primaries
  `IDA_1947_S25F, S2A, S25FFA`.
- **Next:** always -> `sufficiency_evaluator`.

### Node 4 — `sufficiency_evaluator_node` (`sufficiency_node.py`) — LLM call #2
- **Job:** ask "is what we retrieved enough?" and, if not, name the gap.
- **Deterministic guard FIRST (before any LLM call):** if
  `retrieval_iterations >= MAX_ITERATIONS`, returns `sufficient=False,
  missing="iteration_limit_reached"` + warning, **no Gemini call**. Same
  short-circuit if retrieval is empty.
- **Otherwise:** builds a cheap prompt (titles + first 200 chars,
  `PREVIEW_CHARS`, not full text) and calls `_SUFFICIENCY_LLM` (structured ->
  `SufficiencyVerdict{sufficient, missing}`). On LLM error it **degrades
  gracefully**: assumes `sufficient=True` + warning.
- **Writes:** `sufficiency` and increments `retrieval_iterations`.
- **Our run:** `sufficient=True`.
- **Next (conditional):** `_route_after_sufficiency` -> `"end"` -> `answer_synthesis`.

### Node 5 — `graph_expansion_node` (`expansion_node.py`) — deterministic (skipped here)
- **Job:** when evidence is thin, widen the net. MVP strategy: increase
  `max_hops` by 1 (deeper CITES traversal), capped at `MAX_HOPS_CAP=4`.
- **Reads/Writes:** reads `max_hops`/`sufficiency`; writes new `max_hops` +
  warning. If capped, just warns.
- **Next:** always loops back -> `graph_traversal` (re-runs deeper), then
  sufficiency re-evaluates. **This is the one loop in the system.**
- Seen live on the "Section 25N" query: depth went 2 -> 3 because the first pass
  missed 25N.

### Node 6 — `answer_synthesis_node` (`synthesis_node.py`) — LLM call #3
- **Job:** write the human-readable answer, citing only retrieved sections.
- **Deterministic short-circuit FIRST:** if retrieval is empty, returns a fixed
  honest "no applicable sections found" message with **zero citations and no LLM
  call** — the empty path is provably hallucination-free.
- **Otherwise:** builds an **evidence pack** (`_build_human_message`) — full text
  of every retrieved section, each prefixed with `[SECTION_ID]`, plus extracted
  details and the sufficiency verdict — and calls `_SYNTHESIS_LLM` (one retry).
  The prompt (`prompts/synthesis.txt`) orders the structure (concepts ->
  applicable sections -> connected sections -> remedies -> limitations) and
  forbids citing outside the pack.
- **Parses citations:** `_parse_citations` regex-extracts every `[SECTION_ID]`
  marker into `cited_section_ids`. **No verification here** — that is the next
  node's job.
- **`thinking_budget=0` detail:** Gemini 2.5 has "thinking" on by default, and
  those tokens come out of the same `max_output_tokens` budget as the visible
  answer. With a 12-section pack, thinking ate the budget and truncated the
  answer. Setting `thinking_budget=0` (config -> `llm.py`) disables it so the full
  8192 tokens go to the visible, citation-bearing answer.
- **Writes:** `draft_answer` and `cited_section_ids`.
- **Next:** always -> `output_guardrail`.

### Node 7 — `output_guardrail_node` (`output_guardrail_node.py`) — deterministic, hits Neo4j
- **Job:** the **trust boundary**. Enforce (not request) that every citation is
  real, score confidence, inject the disclaimer.
- **Citation verification** (`_verify_citations`): keep an id only if **(a)** it
  is in `state.retrieval.section_ids` (model may only cite what it was *shown*)
  **and (b)** `get_section_by_id(id)` finds it in Neo4j (it actually *exists*).
  Failures are stripped + warned.
- **Confidence** (config weights):
  `0.35*concept_coverage + 0.25*seed_strength + 0.20*sufficiency_score + 0.20*citation_validity`.
  If `< MIN_CONFIDENCE` (0.4), status downgraded to `insufficient_evidence`.
- **Disclaimer:** the `DISCLAIMER` constant is set here — **in code, never by the
  LLM**.
- **Writes:** `verified_section_ids`, `confidence`, `confidence_factors`,
  `status`, `disclaimer`, accumulated `warnings`.
- **Our run:** all 12 citations verified, every factor 1.0, confidence 1.0,
  status `ok`.
- **Next:** always -> `final_response`.

> **Architecture-audit note:** on Neo4j failure this node currently *fails open*
> (falls back to provenance-only, keeps confidence high). The design doc wants
> *fail closed* (cap confidence at 0.3, mark unverified). That is the one change
> still recommended.

### Node 8 — `final_response_node` (`final_response_node.py`) — deterministic, pure formatting
- **Job:** assemble the single user-facing `final_answer` for **every** exit path,
  and never throw (whole body wrapped in try/except returning a safe fallback).
- **Status precedence** (`_resolve_status`): `error` > `rejected` (safety_flag) >
  `out_of_domain` (not in_domain) > guardrail status. This guarantees that even if
  synthesis ran on an out-of-domain query, the output shows the honest "out of
  scope" message and **no legal content**.
- **Citation hygiene:** `_strip_unverified_markers` removes any `[ID]` marker not
  in `verified_section_ids`, so an unverified citation can never appear.
- **For `ok` / `insufficient_evidence`:** cleaned answer + "Verified citations:"
  line + "Confidence:" line with factor breakdown + disclaimer. For
  `out_of_domain` / `rejected` / `error`: fixed honest messages, no legal text.
- **Writes:** `final_answer`, final `status`.
- **Next:** -> `END`. `GRAPH.invoke` returns the final state; `run()` returns it.

That is the full life of one query.

---

<a name="part-5"></a>
## Part 5 — The state object (the clipboard's contents)

Two files, deliberately split:

- **`agents/state.py`** — *domain* dataclasses with **zero external
  dependencies** (no pydantic, no Neo4j, no LangChain). `SectionContext` (one
  legal section + retrieval metadata) and `RetrievalResult` (the bundle of
  sections + edges + confidence, with helpers like `.section_ids` and
  `.primary_sections`). Plain Python so they can be imported anywhere without
  heavy libs. *(It also defines `ExtractedEntities`, `LegalAnswer`,
  `ValidatedAnswer` — older contracts not used by the live LangGraph path; the
  running pipeline uses the pydantic schemas instead.)*

- **`agents/graph_state.py`** — `LegalQueryState`, the **pydantic** orchestration
  state LangGraph passes around. Field groups:
  - input: `raw_query`
  - extraction: `extraction`, `grounded_concepts`
  - retrieval: `retrieval`, `max_hops`, `retrieval_iterations`
  - evaluation: `sufficiency`
  - synthesis: `draft_answer`, `cited_section_ids`
  - step-7 outputs: `verified_section_ids`, `confidence`, `confidence_factors`,
    `status`, `disclaimer`, `final_answer`
  - plus `warnings`, `error`

**Why pydantic for state?** Runtime validation catches a node returning the wrong
shape immediately, and the model serializes cleanly into an API response later.

**How updates merge:** each node returns a dict like `{"retrieval": <obj>}`;
LangGraph overlays those keys onto the current state. For `warnings`, nodes return
`state.warnings + [new]` (read-modify-write) because there is no custom reducer.

---

<a name="part-6"></a>
## Part 6 — Neo4j retrieval, in detail (with Cypher explained)

### Where the connection lives
`graph/db_connection.py`: reads creds from `config.cfg`, builds **one**
`driver = GraphDatabase.driver(uri, auth=(user, pass))` at import, exposes
`get_driver()`. A driver is a connection pool; you open short-lived sessions per
query. Singleton so you are not reconnecting constantly.

### Where the Cypher lives
`graph/queries.py` is the **only** place raw Cypher is written. At **runtime**
only these reads matter:

**1. `get_sections_for_exact_concept(concept_name)`** — the entry point.
```cypher
MATCH (s:Section)-[r:APPLIES_TO]->(c:Concept)
WHERE toLower(c.name) = toLower($concept_name)
RETURN s, c.concept_id AS concept_id, c.name AS concept_name, r.relevance AS relevance
ORDER BY CASE r.relevance WHEN 'primary' THEN 0 ELSE 1 END, s.act_id, s.section_number
```
- `MATCH (s:Section)-[r:APPLIES_TO]->(c:Concept)` — find every "Section connected
  by an APPLIES_TO arrow to a Concept." `s`, `r`, `c` bind to the section, the
  relationship, and the concept.
- `WHERE toLower(c.name) = toLower($concept_name)` — keep matches where the
  concept name equals the one passed (case-insensitive). `$concept_name` is a safe
  parameter.
- `RETURN ...` — hand back the section, the concept id/name, and `r.relevance`
  (the primary/supporting tag stored on the relationship).
- `ORDER BY CASE r.relevance WHEN 'primary' THEN 0 ELSE 1 END, ...` — primaries
  first, then by act and section number.

A looser `get_sections_for_concept` uses `CONTAINS` (substring) for
`main.py` / raw-text callers. The agent path uses the **exact** version because
grounding already produced canonical names.

**2. `get_neighbors(section_id)`** — used during multi-hop expansion.
```cypher
MATCH (s:Section {section_id: $section_id})
CALL (s){
    WITH s MATCH (s)-[r]->(neighbor) RETURN neighbor, type(r) AS rel_type, 'outgoing' AS direction
    UNION
    MATCH (neighbor)-[r]->(s) RETURN neighbor, type(r) AS rel_type, 'incoming' AS direction
}
RETURN neighbor, rel_type, direction
```
- First line locks onto the section with this id (`{section_id: $section_id}` is
  an inline property match).
- `CALL { ... }` is a **subquery**. Inside, two patterns are `UNION`ed: arrows
  going **out** of `s`, and arrows coming **in** to `s`. `type(r)` is the
  relationship type (e.g. `"CITES"`); `'outgoing'`/`'incoming'` tags direction.
- Returns everything one hop away, both directions. Used to walk CITES references.

**3. `get_subgraph(section_ids)`** — assemble the final neighborhood.
```cypher
MATCH (s:Section) WHERE s.section_id IN $section_ids
WITH collect(s) AS sections
OPTIONAL MATCH (source:Section)-[r:CITES]->(target:Section)
WHERE source.section_id IN $section_ids AND target.section_id IN $section_ids
RETURN sections, collect({source_section_id: source.section_id, target_section_id: target.section_id, ...}) AS cites_edges
```
- Grab all sections whose id is in the list (`IN $section_ids`).
- `WITH collect(s) AS sections` — `collect()` rolls those rows into one list;
  `WITH` passes results to the next stage.
- `OPTIONAL MATCH ... CITES ...` — find CITES edges **only between sections
  already in our set**. `OPTIONAL` means "if none exist, do not drop everything."
- Returns sections plus the internal CITES edges, so the evidence pack knows which
  retrieved sections cite each other.

**4. `get_section_by_id(section_id)`** — the verification query used by the guardrail.
```cypher
MATCH (s:Section {section_id: $section_id}) RETURN s LIMIT 1
```
Does a section with this exact id exist? Returns it or `None`. This is the
"exists in Neo4j" half of citation verification — a **deterministic Cypher
lookup, not an LLM judgment**. That is the whole anti-hallucination thesis.

### The traversal orchestration (`graph/traversal.py`)
`traverse(concept_name, max_hops, max_sections, exact)` is a **breadth-first
search (BFS)**:

1. **Find anchors** (`_find_anchors`): call `get_sections_for_exact_concept` ->
   split into primary and supporting sections (trusted entry points via
   APPLIES_TO).
2. **Score confidence** (`_score_confidence`): `1.0` if any primary anchors,
   `0.6` if only supporting, `0.0` if none. (Becomes `seed_strength` later.)
3. **If no anchors:** return empty (concept is not in the graph).
4. **Expand** (`_expand_from_anchors`): BFS outward from primary anchors over
   CITES edges, up to `max_hops` rounds. Keeps `visited` + `frontier`; each hop
   fetches neighbors via `get_neighbors`, filters to Sections, collects new ones.
   Stops early if a hop finds nothing.
5. **Assemble + dedup + cap:** combine anchors + expanded, dedup by id; if total >
   `max_sections` (15), keep all anchors and trim expanded (anchors higher-trust).
6. **Fetch the subgraph** (`get_subgraph`) to attach CITES edges among the final
   set.
7. Return a `TraversalResult` (sections, edges, hops taken, confidence, is_empty).

Plain English: *start at the sections the concept points to, follow citation
links outward up to N hops, dedup, cap the size, report how many hops and how
strong the entry point was.*

### How results flow back into state
`retrieval_node.py` converts each raw Neo4j dict into a typed `SectionContext`
(via `SectionContext.from_dict`, which handles missing keys defensively), merges
across concepts, and packs everything into a `RetrievalResult` written to
`state.retrieval`.

---

<a name="part-7"></a>
## Part 7 — The tunable parameters

All live in **`config.py`** (pydantic-settings; any field overridable by an env
var of the same name). One lives in `ontology.py`.

| Parameter | Default | Where used | Controls | Raise it -> | Lower it -> |
|---|---|---|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | `llm.py` | Which model all 3 calls use | (swap to `pro`) higher quality, slower, costlier | n/a |
| `EXTRACTION_TEMPERATURE` / `SUFFICIENCY_TEMPERATURE` | `0.0` | those nodes | Randomness of structured calls | more erratic extraction | already at floor |
| `SYNTHESIS_TEMPERATURE` | `0.2` | synthesis | Creativity of the prose | more fluent, more drift risk | more rigid, more faithful |
| `SYNTHESIS_MAX_TOKENS` | `8192` | synthesis | Max answer length | longer answers, costlier | truncation risk on big packs |
| `SYNTHESIS_THINKING_BUDGET` | `0` | synthesis | Gemini "thinking" tokens | thinking eats answer budget -> truncation | 0 = off; max visible answer |
| `MAX_RETRIEVAL_ITERATIONS` | `2` | `sufficiency_node` + router | Extra retrieval loops allowed | more chances, slower/costlier | gives up faster |
| `MAX_SECTIONS` | `15` | `retrieval_node`/`traversal` | Cap on sections per traversal | richer but pricier/slower synthesis | leaner, may miss law |
| `MAX_HOPS_DEFAULT` | `2` | initial `max_hops` | Starting CITES depth | wider first net, more noise | narrower, may miss connected sections |
| `MAX_HOPS_CAP` | `4` | `expansion_node` | Ceiling expansion reaches | deeper loops possible | expansion does little |
| `LANGGRAPH_RECURSION_LIMIT` | `25` | `run()` invoke | Hard stop on total node steps | safer vs runaway | could abort legit long runs |
| `SUFFICIENCY_PREVIEW_CHARS` | `200` | `sufficiency_node` | Text shown to sufficiency LLM | better-informed verdict, costlier | cheaper, blunter |
| `CONFIDENCE_W_*` | `0.35/0.25/0.20/0.20` | `output_guardrail` | Weighting of the 4 factors | emphasizes that factor | de-emphasizes it (must sum to 1.0) |
| `MIN_CONFIDENCE` | `0.4` | `output_guardrail` | Threshold for `insufficient_evidence` | stricter — more flagged weak | lenient — more pass as `ok` |
| `FUZZY_CUTOFF` | `0.8` | `ontology.py` `ground_query` | How close a fuzzy match must be | fewer false matches, more misses | more matches, more wrong groundings |

**The four confidence factors** (in `output_guardrail_node`, each in [0,1]):
- `concept_coverage` = grounded / extracted concepts -> *did we recognize the question?*
- `seed_strength` = traversal confidence (1.0 primary / 0.6 supporting / 0.0 none)
  -> *how strong was the graph entry point?*
- `sufficiency_score` = 1.0 sufficient / 0.5 partial / 0.0 none -> *did the LLM
  think we had enough?*
- `citation_validity` = verified / cited -> *did the model cite real, shown sections?*

**Interview framing:** the confidence score is not a vibe from the LLM — it is a
transparent weighted sum of four measurable signals, each logged in
`confidence_factors`. If an answer scores low, the lowest factor names the failing
stage: low `concept_coverage` -> grounding/ontology gap; low `seed_strength` ->
concept maps to too few sections; low `sufficiency_score` -> genuine graph
coverage gap; low `citation_validity` -> synthesis prompt let the model drift.

---

<a name="part-8"></a>
## Part 8 — Tracing & observability (the honest state of it)

**What exists today:**
1. **Python `logging`** in the nodes (`logger.info/warning/exception` in
   `retrieval_node`, `sufficiency_node`, `synthesis_node`, `output_guardrail_node`,
   `traversal.py`). E.g. the guardrail logs
   `"%s/%s citations verified, confidence=%.2f, status=%s"`. This is the real,
   working observability. Set verbosity with `logging.basicConfig(level=...)`.
2. **In-band trace via state** — the `warnings` list and `confidence_factors`
   dict travel *inside* the response. Every degradation (stripped citation, Neo4j
   fallback, expansion, sufficiency-LLM failure) appends a readable warning. The
   answer carries its own audit trail.
3. **`_print_result` in `graph_agent.py`** — prints status, confidence, factors,
   cited vs verified ids, warnings, and the final answer. The manual inspection
   tool.

**What is configured but not actively wired:**
- LangSmith env vars exist (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`,
  `LANGCHAIN_PROJECT`) and `config.py` carries them. **Nuance:** LangChain /
  LangGraph emit traces to LangSmith **automatically** if
  `LANGCHAIN_TRACING_V2=true` and the API key are set — because the LLM calls go
  through LangChain's `ChatVertexAI` and the pipeline runs through LangGraph, both
  auto-instrumented. There is **no explicit tracing code** in the repo; it is
  "ambient." Honest statement: *LangSmith tracing is wired by configuration, not
  by code — set the env flag and traces flow; explicit run-id capture and
  `@traceable` decorators on the Neo4j functions are not yet added.* The
  architecture doc (§12) describes the fuller intended setup — roadmap, not built.

**How you would read it:** with the env flag on, each `GRAPH.invoke` appears in
the LangSmith UI as a tree — the 8 node spans, the 2–3 Gemini calls with
prompts/outputs/token counts, and the traversal calls. Without it, rely on the
console logs + `warnings`/`confidence_factors` in the returned state.

---

<a name="part-9"></a>
## Part 9 — The architecture, top to bottom

(See `ARCHITECTURE_DIAGRAM.md` for the full diagrams.)

**The layers, top to bottom:**
1. **Entry** — `graph_agent.py` (`run()` -> `GRAPH.invoke`).
2. **Orchestration** — the compiled LangGraph: 8 nodes, 1 conditional fork, 1
   loop. The single source of control flow.
3. **State** — `LegalQueryState` (pydantic, `graph_state.py`) + the
   dependency-free domain models in `state.py`.
4. **LLM layer** — `llm.py` builds one `ChatVertexAI` per role; prompts as files
   in `agents/prompts/`; structured output via pydantic schemas in `schemas.py`.
   Auth is ADC (not a service-account key file — a deliberate divergence from the
   design doc, documented in `llm.py`).
5. **Graph layer** — `db_connection.py` (driver), `queries.py` (all Cypher),
   `traversal.py` (BFS). Read-only at runtime.
6. **Knowledge layer** — Neo4j, populated **offline** by `ingest/` (PDF ->
   sections.jsonl / relationships.jsonl -> `graph_builder.py` loads nodes+edges ->
   `ontology_loader.py` loads the 25 concepts + APPLIES_TO). `ontology.py` mirrors
   that concept map for in-process grounding.
7. **Config** — `config.py` centralizes every tunable; `.env` supplies secrets.

**The one-sentence thesis:** *Every section the user sees was reached
deterministically by walking the Neo4j graph from a grounded concept, and
re-verified to exist in Neo4j before display — the LLM only phrases the answer, it
never chooses or invents the law.* The three LLM calls (understand the question,
judge sufficiency, write the prose) are bracketed on both sides by deterministic
graph operations, and the only loop is governed by plain-Python counters, not the
model.

---

## Quick file map

| File | Role |
|---|---|
| `graph_agent.py` | Entry point — builds/compiles the graph, `run()` |
| `agents/graph_state.py` | `LegalQueryState` — the shared state |
| `agents/state.py` | Dependency-free domain dataclasses |
| `agents/schemas.py` | Pydantic schemas for structured LLM output |
| `agents/llm.py` | `ChatVertexAI` factory (reads config) |
| `agents/ontology.py` | Deterministic concept grounding |
| `agents/nodes/extraction_node.py` | LLM #1 — entity/concept extraction |
| `agents/nodes/grounding_node.py` | Deterministic grounding |
| `agents/nodes/retrieval_node.py` | Deterministic Neo4j retrieval |
| `agents/nodes/sufficiency_node.py` | LLM #2 — sufficiency verdict |
| `agents/nodes/expansion_node.py` | Deterministic loop expansion |
| `agents/nodes/synthesis_node.py` | LLM #3 — answer synthesis |
| `agents/nodes/output_guardrail_node.py` | Deterministic — verify + confidence + disclaimer |
| `agents/nodes/final_response_node.py` | Deterministic — assemble final answer |
| `graph/db_connection.py` | Neo4j driver singleton |
| `graph/queries.py` | All Cypher (read + write) |
| `graph/traversal.py` | BFS traversal engine |
| `graph/schema.py` | Node/relationship label definitions |
| `config.py` | All runtime configuration |
| `main.py` | Legacy CLI traversal tester (no LLM) |
| `ingest/graph_builder.py` | Offline loader: JSONL -> Neo4j |
