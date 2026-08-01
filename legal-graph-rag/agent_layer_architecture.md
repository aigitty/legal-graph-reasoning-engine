# Legal Graph RAG — Agent Layer Architecture (v1)

> **HISTORICAL DESIGN DOCUMENT — for reference only.**
> This was the architecture plan written before implementation. The system is now
> fully built. For the current state of the project, read `CLAUDE.md` (rules +
> status), `HOW_IT_WORKS_SIMPLE.md` (plain-English walkthrough), and
> `SYSTEM_EXPLAINED.md` (deep technical walkthrough). Key divergences from this
> plan: the folder is `agents/` not `agent/`; FastAPI schemas use `persona` not
> `langsmith_run_id`/`timings_ms`; auth is ADC not a service-account key file.

Design document for the redesigned reasoning/agent layer, built on top of the completed ingestion layer (5 Acts, 255 Section nodes, 127 CITES edges, 25 Concept nodes, 75 APPLIES_TO edges, `graph/traversal.py` deterministic retrieval).

Stack: **LangGraph (orchestration) + LangChain (model/tool layer) + Gemini via Vertex AI (LLM) + Neo4j (graph) + FastAPI (local runtime) + LangSmith (observability)**.

---

## 1. Executive Summary

**Recommendation: a single LangGraph state-machine workflow with ~9 specialized nodes. Not multi-agent for MVP.**

The system is a **pipeline with conditional loops**, not a negotiation between autonomous agents. Every step has a fixed responsibility, a fixed contract (state in → state out), and a deterministic or single-LLM-call implementation. There is exactly one decision-driven loop: *"is the retrieved graph evidence sufficient? If not, expand traversal and retry."* That is a conditional edge, not an agent conversation.

Why LangGraph is the right orchestration layer (and a plain LangChain chain is not):

1. **Conditional branching** — out-of-domain queries exit early; insufficient evidence loops back to expansion; failed citation validation degrades the answer instead of shipping it. Linear chains can't express this cleanly.
2. **Shared typed state** — one `LegalQueryState` object flows through all nodes, so the final answer can always be traced back to grounded concepts → seed sections → traversal results → validated citations. This *is* the "reasoned legal path" output requirement.
3. **Loop control with hard limits** — max-iteration / max-depth guards live in graph edges, not buried in prompt instructions.
4. **Checkpointing/extension later** — v2 ruling retrieval or a multi-agent split becomes "add a node/subgraph," not a rewrite.

The LLM (Gemini via Vertex AI) is called in exactly **three** places: entity/concept/jurisdiction extraction, sufficiency evaluation, and answer synthesis. Everything else — grounding, traversal, citation validation, confidence scoring, disclaimers — is deterministic Python + Cypher. The LLM never invents a section; it only describes sections the graph returned.

---

## 2. Agent Layer Responsibility

**The agent layer owns:**

* Accepting a plain-language legal query (via FastAPI).
* Validating that the query is in-domain (Indian employment law, MVP scope) and safe.
* Extracting structured intent: legal concepts, entities (employer type, tenure, event), jurisdiction (state).
* Grounding extracted concepts against the **existing 25-concept ontology** (no new concepts invented at runtime).
* Calling the deterministic Neo4j traversal (Concept → APPLIES_TO → Section → CITES) with controlled parameters.
* Deciding whether retrieved evidence is sufficient, and expanding traversal (bounded) if not.
* Synthesizing a grounded, human-readable answer **only from retrieved section text**.
* Validating every cited section against Neo4j before returning.
* Computing a confidence score and attaching guardrails (disclaimer, insufficient-evidence honesty).
* Emitting LangSmith traces and returning a trace/run ID to the caller.

**The agent layer does NOT own:**

* PDF parsing, section extraction, relationship extraction (done: `ingest/`).
* Graph construction, schema, ontology loading (done: `graph/schema.py`, `ingest/graph_builder.py`, `ingest/ontology_loader.py`).
* Modifying the graph at runtime — the agent layer is **read-only** against Neo4j.
* Ruling ingestion or INTERPRETS logic (v2).
* Deployment, hosting, CI/CD (explicitly out of scope).
* Long-term conversation memory (not needed for MVP; one query → one reasoned answer).

---

## 3. Recommended Runtime Architecture

```
FastAPI (local)  →  LangGraph workflow  →  { Neo4j retrieval layer, Gemini/Vertex AI layer, Guardrails }  →  LangSmith
```

**FastAPI local runtime layer (`api/`)**
Thin HTTP wrapper. One main endpoint (`POST /query`) that builds the initial state, invokes the compiled LangGraph app, and serializes the final state into a response schema. No business logic lives here.

**LangGraph workflow (`agent/workflow.py`)**
The compiled `StateGraph` — the single source of truth for control flow. Nodes are small functions; edges encode all branching/looping. This is the "agent."

**LangChain tool/model layer (`agent/llm.py`, `agent/schemas.py`)**
One configured `ChatVertexAI` client, wrapped with `.with_structured_output(PydanticSchema)` for extraction and sufficiency, and plain text output for synthesis. Prompts live in `agent/prompts/` as versioned files.

**Neo4j graph access layer (`graph/`)**
Existing `db_connection.py` + `queries.py`, plus a refactored `graph/retrieval.py` (evolved from `traversal.py`, see §8) exposing two pure functions to the agent: `retrieve(concepts, jurisdiction, depth, top_k)` and `verify_sections(section_ids)`. The agent never writes Cypher inline.

**Gemini via Vertex AI LLM layer**
`langchain-google-vertexai`'s `ChatVertexAI`, authenticated with your existing service account (`GOOGLE_APPLICATION_CREDENTIALS`). Model name, temperature, and max tokens come from `config.py` so you can swap Flash/Pro without code changes.

**Guardrails layer (`agent/guardrails.py`)**
Two halves: input guardrails (domain/safety classification — piggybacked onto the extraction LLM call to save a call) and output guardrails (deterministic citation verification against Neo4j, confidence thresholding, mandatory disclaimer injection).

**LangSmith observability layer**
Enabled via environment variables (`LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=legal-graph-rag`). Every node, LLM call, and the traversal tool call appear as spans automatically because everything runs inside the LangGraph invocation. The run ID is captured and returned in the API response.

---

## 4. Single-Agent vs Multi-Agent Decision

**Decision: single LangGraph workflow. Multi-agent is not justified for this MVP.**

Multi-agent designs earn their complexity when (a) subtasks need *different tools, different context windows, or genuinely independent reasoning loops*, (b) tasks run in parallel over heterogeneous sources, or (c) agents must critique/negotiate with each other. None of these hold here:

* The corpus is one Neo4j graph with 255 sections. There is one retrieval tool.
* Extraction, sufficiency, and synthesis are each **one structured LLM call**, not open-ended reasoning loops. Wrapping a single call in an "agent" adds prompt overhead, latency, more failure modes, and harder LangSmith traces — for zero capability gain.
* The "guardrails agent" pattern is strictly worse than deterministic validation here: citation existence is a Cypher query, not a judgment call. An LLM judging whether citations exist is exactly the hallucination surface you're trying to eliminate.

**Trade-off acknowledged:** a multi-agent build would give you more exposure to LangGraph's supervisor/handoff patterns. But it would teach the *wrong* lesson for Graph RAG — that orchestration complexity substitutes for retrieval quality. You get the educational depth you want from: typed state design, conditional edges, bounded loops, structured outputs, and tracing. That is the 80% of real LangGraph engineering.

**Evolution path (so nothing is closed off):** each node already has a single responsibility and a state contract. To go multi-agent in v2, you promote a node to a subgraph:

* `graph_traversal_node` → a *Retrieval subgraph* that fans out over statutes and rulings in parallel when INTERPRETS edges exist.
* `answer_synthesis_node` → a *Reasoner + Critic* pair if synthesis quality needs adversarial checking.
* The top-level graph becomes a supervisor routing between subgraphs. State schema barely changes.

---

## 5. LangGraph State Design

**Use Pydantic `BaseModel`** (not TypedDict). Reasons: runtime validation catches node contract bugs early; the same models serialize directly into FastAPI responses; nested models (sections, citations) stay typed. LangGraph supports Pydantic state natively.

```python
# agent/state.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ExtractedEntities(BaseModel):
    employment_type: Optional[str] = None      # "private company", "factory", ...
    tenure_years: Optional[float] = None
    triggering_event: Optional[str] = None     # "terminated without notice"
    monetary_claim: Optional[str] = None       # "unpaid wages", "gratuity", ...

class RetrievedSection(BaseModel):
    section_id: str            # e.g. "IDA_1947/25F" — must match Neo4j key
    act: str
    section_number: str
    title: str
    text: str
    source: Literal["seed", "cites_hop1", "cites_hop2"]
    score: float
    matched_concepts: list[str] = []

class Citation(BaseModel):
    section_id: str
    verified: bool = False     # set ONLY by output guardrail via Neo4j lookup

class LegalQueryState(BaseModel):
    # input
    raw_query: str
    normalized_query: str = ""

    # extraction
    entities: Optional[ExtractedEntities] = None
    jurisdiction: Optional[str] = None              # e.g. "Karnataka"
    legal_concepts: list[str] = []                  # free-form, from LLM
    grounded_concepts: list[str] = []               # subset matching the 25-concept ontology
    in_domain: bool = True
    safety_flag: Optional[str] = None

    # retrieval
    seed_sections: list[str] = []                   # section_ids from APPLIES_TO
    traversal_results: list[RetrievedSection] = []
    traversal_depth: int = 1
    retrieval_iterations: int = 0                   # loop guard

    # evaluation & synthesis
    sufficiency: Literal["pending", "sufficient", "insufficient", "exhausted"] = "pending"
    sufficiency_reason: str = ""
    draft_answer: str = ""
    citations: list[Citation] = []
    confidence: float = 0.0
    confidence_factors: dict[str, float] = {}

    # control & observability
    errors: list[str] = []
    warnings: list[str] = []
    final_answer: str = ""
    status: Literal["ok", "insufficient_evidence", "out_of_domain", "rejected", "error"] = "ok"
    langsmith_run_id: Optional[str] = None
    timings_ms: dict[str, int] = {}
```

Notes:
* `retrieval_iterations` and `traversal_depth` are the loop-control fields — they live in state so LangSmith traces show exactly why the loop stopped.
* `Citation.verified` defaults to `False` and only the deterministic guardrail can flip it. Synthesis output that cites an unverified ID never reaches the user.

---

## 6. LangGraph Nodes

All nodes take `LegalQueryState` and return a partial update. **D** = deterministic, **LLM** = one Gemini call.

### 1. `input_guardrail_node` — D (with LLM-assisted domain signal, see note)
* **Purpose:** normalize the query (strip, collapse whitespace, length limits), run cheap deterministic checks (empty query, >2,000 chars, obvious injection patterns), and set early-exit flags.
* **In:** `raw_query` → **Out:** `normalized_query`, possibly `status="rejected"`, `errors`.
* **Failure:** sets `status="rejected"` and routes straight to `final_response_node`. Never throws.
* **Note:** out-of-domain detection requires understanding, so the *domain verdict itself* comes from the extraction call (next node) — one LLM call does double duty. This node handles only what regex/length checks can decide.

### 2. `entity_extraction_node` — **LLM** (Gemini, structured output)
* **Purpose:** one Vertex AI call with `.with_structured_output(ExtractionResult)` extracting: `legal_concepts` (free-form), `entities`, `jurisdiction`, `in_domain: bool`, `safety_flag`. Prompt explicitly lists MVP scope ("Indian employment/labour law") so `in_domain` is meaningful.
* **In:** `normalized_query` → **Out:** `legal_concepts`, `entities`, `jurisdiction`, `in_domain`, `safety_flag`.
* **Failure:** on parse/API error, retry once; then set `errors`, `status="error"` → final response with a graceful failure message.

### 3. `concept_grounding_node` — **D**
* **Purpose:** map free-form `legal_concepts` to the **25 canonical Concept nodes**. Strategy: (a) exact/normalized string match against concept names, (b) alias table loaded from `concept_map.json` synonyms (extend the JSON with an `aliases` field — e.g. "fired", "sacked", "terminated" → `wrongful_termination`), (c) token-overlap fuzzy fallback (e.g. `rapidfuzz` ratio ≥ 85). No embeddings needed for 25 concepts.
* **In:** `legal_concepts` → **Out:** `grounded_concepts`, `warnings` (for concepts that didn't ground).
* **Failure:** zero grounded concepts → `status="insufficient_evidence"` path via conditional edge ("the graph has no entry point for this query") — the system says so instead of guessing.

### 4. `graph_traversal_node` — **D**
* **Purpose:** call `graph/retrieval.py::retrieve(grounded_concepts, jurisdiction, depth=state.traversal_depth, top_k=12)`. Returns seed sections (APPLIES_TO) plus CITES expansion, scored and deduplicated (§8).
* **In:** `grounded_concepts`, `jurisdiction`, `traversal_depth` → **Out:** `seed_sections`, `traversal_results` (merged/deduped across iterations), `retrieval_iterations += 1`.
* **Failure:** Neo4j connection error → one retry, then `status="error"`. Empty results → handled by sufficiency node, not an error.

### 5. `sufficiency_evaluator_node` — **LLM** (Gemini, structured output, cheap)
* **Purpose:** given the query, entities, and *titles + first ~300 chars* of retrieved sections (not full text — keep it cheap), return `{sufficient: bool, missing: str}`. Examples of "insufficient": user asks about gratuity but no Payment of Gratuity Act sections retrieved; jurisdiction is Karnataka but only central-act sections present.
* **In:** `normalized_query`, `entities`, `traversal_results` → **Out:** `sufficiency`, `sufficiency_reason`.
* **Hard rule:** if `retrieval_iterations >= MAX_ITERATIONS (2)`, this node sets `sufficiency="exhausted"` *deterministically before any LLM call* — the LLM never controls loop termination.
* **Failure:** LLM error → treat current evidence as sufficient if non-empty (degrade gracefully), else `exhausted`; add a `warning`.

### 6. `graph_expansion_node` — **D**
* **Purpose:** bounded expansion strategy, in order: (1) raise `traversal_depth` 1 → 2 (one extra CITES hop); (2) widen `top_k`; (3) add un-grounded-but-near-miss concepts from grounding (fuzzy score 70–85) as secondary seeds. Exactly one strategy per loop pass.
* **In:** `sufficiency_reason`, `traversal_depth`, `retrieval_iterations` → **Out:** updated `traversal_depth` / retrieval params, `warnings`.
* **Failure:** if no strategy remains, set `sufficiency="exhausted"` so the loop exits.

### 7. `answer_synthesis_node` — **LLM** (Gemini)
* **Purpose:** the main reasoning call. Input is an **evidence pack**: query, entities, jurisdiction, and the full text of top retrieved sections, each tagged with its `section_id`. Prompt rules: *cite only provided `section_id`s, in the form `[IDA_1947/25F]`; if evidence doesn't cover part of the question, say so explicitly; structure output as: concepts detected → applicable sections & why → connected sections → remedies/conditions → limitations.* If `sufficiency="exhausted"`, the prompt instead instructs a partial/insufficient-evidence answer.
* **In:** `normalized_query`, `entities`, `jurisdiction`, `traversal_results`, `sufficiency` → **Out:** `draft_answer`, `citations` (parsed from `[...]` markers, all `verified=False`).
* **Failure:** retry once; then `status="error"` with graceful message.

### 8. `output_guardrail_node` — **D**
* **Purpose:** the trust boundary. Steps: (1) **citation existence check** — `verify_sections([c.section_id ...])` runs one Cypher `MATCH` against Neo4j; any citation not found is stripped from the answer and logged as a warning; (2) **retrieval-set check** — every citation must also be in `traversal_results` (the LLM may only cite what it was shown); (3) **confidence scoring** — deterministic composite (see formula below); (4) **threshold** — confidence < 0.4 → downgrade `status` to `insufficient_evidence` and replace prescriptive language with hedged language; (5) **disclaimer** — appended in code, never left to the LLM.
* **Confidence formula (transparent, logged in `confidence_factors`):**
  `confidence = 0.35·concept_coverage + 0.25·seed_strength + 0.20·sufficiency_score + 0.20·citation_validity`
  where concept_coverage = grounded/extracted concepts; seed_strength = min(1, seeds/3); sufficiency_score = 1.0 sufficient / 0.4 exhausted; citation_validity = verified/total citations.
* **In:** `draft_answer`, `citations`, plus scoring inputs → **Out:** `citations` (verified flags), `confidence`, `confidence_factors`, possibly modified `draft_answer`, `warnings`.
* **Failure:** Neo4j unreachable during verification → **fail closed**: mark all citations unverified, cap confidence at 0.3, add prominent warning.

### 9. `final_response_node` — **D**
* **Purpose:** assemble `final_answer` for every exit path (success, insufficient evidence, out-of-domain, rejected, error). Formats the reasoned legal path (concepts → sections → connections → remedies → confidence → disclaimer). Records `langsmith_run_id` and `timings_ms`.
* **In:** everything → **Out:** `final_answer`, final `status`.
* **Failure:** none — pure formatting; it is the safety net node.

---

## 7. Conditional Edges and Looping Logic

```
START
  │
  ▼
input_guardrail_node ──[rejected]──────────────────────────┐
  │ ok                                                     │
  ▼                                                        │
entity_extraction_node ──[error]───────────────────────────┤
  │ ok                                                     │
  ├──[in_domain == False or safety_flag]───────────────────┤
  ▼                                                        │
concept_grounding_node                                     │
  │                                                        │
  ├──[grounded_concepts empty]──── (insufficient path) ────┤
  ▼                                                        │
graph_traversal_node                                       │
  │                                                        │
  ▼                                                        │
sufficiency_evaluator_node                                 │
  │                                                        │
  ├──[insufficient AND iterations < 2]──► graph_expansion_node
  │                                              │         │
  │                  ┌───────────────────────────┘         │
  │                  ▼  (loops back)                       │
  │            graph_traversal_node                        │
  │                                                        │
  ▼ [sufficient OR exhausted]                              │
answer_synthesis_node                                      │
  │                                                        │
  ▼                                                        │
output_guardrail_node                                      │
  │                                                        │
  ▼                                                        ▼
final_response_node ◄──────────────────────────────────────┘
  │
  ▼
 END
```

* **Loop:** only one — `sufficiency → expansion → traversal → sufficiency`.
* **Hard limits:** `MAX_ITERATIONS = 2` extra retrieval passes (3 traversals total worst case); `MAX_DEPTH = 2` CITES hops; `TOP_K_CAP = 20` sections in the evidence pack. All enforced in deterministic code. Also set LangGraph's `recursion_limit` (~25) as a belt-and-braces guard.
* **Early exits:** rejected input, out-of-domain, unsafe request, extraction error, zero grounded concepts — all jump directly to `final_response_node` with the appropriate status. No LLM synthesis happens on these paths (cheaper, safer).
* **Exhausted ≠ error:** when the loop exhausts, synthesis still runs but produces an honest partial answer ("the graph contains X and Y, but does not cover Z").

---

## 8. Deterministic Graph Retrieval Design

**Refactor `graph/traversal.py` → `graph/retrieval.py`** (keep the old file until parity is tested). The existing logic — concept-seeded traversal, CITES expansion, dedupe, result limiting — is the right core. Refactor it into a parameterized, agent-callable API:

```python
def retrieve(concepts: list[str], jurisdiction: str | None,
             depth: int = 1, top_k: int = 12) -> RetrievalResult: ...

def verify_sections(section_ids: list[str]) -> dict[str, bool]: ...
```

**Traversal shape (one Cypher pattern per stage, not one mega-query):**

1. **Seeds:** `MATCH (c:Concept)-[:APPLIES_TO]->(s:Section) WHERE c.name IN $concepts` — these are the highest-trust sections.
2. **Expansion:** from seeds, `MATCH (s)-[:CITES*1..$depth]->(t:Section)` (and the incoming direction `(t)-[:CITES]->(s)` at depth 1 — a section *cited by* your seed is often the operative provision). Tag each result with its hop distance.
3. **Dedup:** by `section_id`; if a section is reached multiple ways, keep the best (lowest hop, most matched concepts) and merge `matched_concepts`.

**Scoring (deterministic, explainable):**

```
score = 2.0 · (# matched concepts)            # multi-concept sections are the heart of the answer
      + 1.5 · (1 if seed else 0)
      + 1.0 / (1 + hop_distance)              # decay over CITES hops
      + 0.5 · (1 if act matches jurisdiction) # e.g. Karnataka Shops Act for Karnataka queries
```

Sort by score, take `top_k`. With 255 sections and 127 edges, this runs in milliseconds and is fully auditable in LangSmith.

**Why retrieval must not rely on LLM judgment:** the graph encodes legal structure that the LLM cannot verify — that §25F is what "retrenchment" actually points to, that §25F is conditioned by §25B. If the LLM selected sections, you'd be back to vector-RAG-with-extra-steps: plausible-sounding but unverifiable. Deterministic traversal makes every retrieved section *provably reachable* from a grounded concept, which is the entire identity of this project. The LLM's only retrieval-adjacent job is judging *whether the deterministic result set answers the question* — and even that judgment can only trigger a deterministic, bounded expansion.

---

## 9. Gemini / Vertex AI Usage Design

**Exactly three call sites** (plus zero hidden ones — guardrails and grounding are code):

| Call site | Model | Output mode | Temp |
|---|---|---|---|
| Entity/concept/jurisdiction extraction (+ domain & safety verdict) | `gemini-2.5-flash` | `.with_structured_output(ExtractionResult)` | 0.0 |
| Sufficiency evaluation | `gemini-2.5-flash` | `.with_structured_output(SufficiencyVerdict)` | 0.0 |
| Answer synthesis | `gemini-2.5-flash` for MVP; one-line config switch to `gemini-2.5-pro` if synthesis quality disappoints | text | 0.2 |

(Pin whatever exact model IDs your Vertex AI region exposes in `config.py`; treat the table as the tiering decision — cheap/fast structured calls, optionally stronger synthesis.)

**LangChain integration:**

```python
# agent/llm.py
from langchain_google_vertexai import ChatVertexAI
from config import settings

def get_llm(role: str) -> ChatVertexAI:
    return ChatVertexAI(
        model_name=settings.models[role],          # "extraction" | "sufficiency" | "synthesis"
        project=settings.gcp_project,
        location=settings.gcp_location,
        temperature=settings.temps[role],
        max_output_tokens=settings.max_tokens[role],
    )

extraction_llm = get_llm("extraction").with_structured_output(ExtractionResult)
```

Auth: your existing service account via `GOOGLE_APPLICATION_CREDENTIALS` env var — no extra cloud architecture needed.

**Pydantic schemas for structured calls:**

```python
class ExtractionResult(BaseModel):
    legal_concepts: list[str] = Field(description="Plain legal concepts implied by the query")
    jurisdiction: Optional[str] = Field(description="Indian state if stated/implied, else null")
    entities: ExtractedEntities
    in_domain: bool = Field(description="True only if Indian employment/labour law")
    safety_flag: Optional[str] = Field(description="Set if request seeks to evade law/harm someone")

class SufficiencyVerdict(BaseModel):
    sufficient: bool
    missing: str = Field(description="What legal coverage is missing, if any")
```

**Keeping Gemini grounded (the four mechanisms):**

1. **Evidence-pack-only synthesis** — the synthesis prompt contains the full text of retrieved sections with IDs; the system instruction forbids citing anything outside the pack and forbids stating legal rules not present in the pack.
2. **Citation marker contract** — every legal claim must carry a `[section_id]` marker; markers are parsed and verified post-hoc.
3. **Deterministic post-verification** — `output_guardrail_node` strips any citation that fails the Neo4j check or wasn't in the evidence pack (§6.8). Grounding is *enforced*, not requested.
4. **Honest-insufficiency prompting** — when `sufficiency != "sufficient"`, the prompt explicitly instructs a partial answer naming the gap, and the deterministic layer caps confidence.

---

## 10. Guardrails Design

**Input guardrails**

| Check | Mechanism | On failure |
|---|---|---|
| Empty / oversized / garbage query | Deterministic (length, charset) | `status="rejected"`, polite message, no LLM call |
| Out-of-domain (criminal law, taxes, non-Indian law, non-legal) | `in_domain` flag from extraction call | `status="out_of_domain"`; reply states MVP scope (Indian employment law) and lists what it *can* answer |
| Unsafe request (evading legal obligations, harming an employee, fabricating evidence) | `safety_flag` from extraction call + a small deterministic keyword screen | `status="rejected"` with a brief refusal; never reaches retrieval |
| Missing critical facts (e.g., no tenure, no state) | Deterministic check on `entities` | **Do not block.** Proceed, record assumptions as `warnings`, and surface them in the answer ("Assuming a private establishment; if you work in a factory, the Industrial Disputes Act applies differently"). Blocking on missing facts would make the MVP unusable. |

**Output guardrails**

| Check | Mechanism | On failure |
|---|---|---|
| Citation existence | Cypher `MATCH (s:Section) WHERE s.section_id IN $ids` | Strip unverified citation + its sentence-level claim; log warning; reduce `citation_validity` factor |
| Citation provenance | Citation ∈ `traversal_results` | Same as above (catches the LLM citing a real section it wasn't shown) |
| Section-text consistency (MVP-light) | Deterministic: cited section's `matched_concepts`/act must relate to the claim context; full LLM-as-judge consistency check deferred to v2 | Add warning, small confidence penalty |
| Confidence threshold | Composite score (§6.8) < 0.4 | `status="insufficient_evidence"`; answer reframed as "possibly relevant provisions" rather than conclusions |
| Legal-advice disclaimer | **Appended in code, always**, in `final_response_node` | n/a — cannot be skipped |

**Validation-failure philosophy:** degrade honestly rather than block. A stripped-citation answer with a warning and lower confidence is more useful (and more trustworthy) than a refusal — *except* for safety flags and out-of-domain, which exit before any legal content is generated.

---

## 11. FastAPI Local Runtime Design

**Endpoints (local only):**

```
POST /query           # main reasoning endpoint
GET  /health          # checks Neo4j connectivity + config sanity
GET  /graph/stats     # node/edge counts (debugging aid, reuses queries.py)
GET  /concepts        # the 25 grounded concepts (useful for testing & UI later)
```

**Schemas:**

```python
class QueryRequest(BaseModel):
    query: str
    jurisdiction_hint: Optional[str] = None       # optional override
    max_sections: int = 12

class SectionCitation(BaseModel):
    section_id: str
    act: str
    section_number: str
    title: str
    verified: bool

class QueryResponse(BaseModel):
    status: Literal["ok", "insufficient_evidence", "out_of_domain", "rejected", "error"]
    answer: str
    detected_concepts: list[str]
    grounded_concepts: list[str]
    jurisdiction: Optional[str]
    citations: list[SectionCitation]
    confidence: float
    confidence_factors: dict[str, float]
    warnings: list[str]
    langsmith_run_id: Optional[str]
    timings_ms: dict[str, int]
```

**Trace ID return:** generate a UUID per request, pass it as the LangSmith run ID, and echo it back:

```python
import uuid
run_id = uuid.uuid4()
result = await app_graph.ainvoke(initial_state, config={"run_id": run_id, "metadata": {...}})
response.langsmith_run_id = str(run_id)
```

This gives you a copy-paste handle from any API response straight into the LangSmith UI.

**Sync vs async: use `async def` endpoints with `graph.ainvoke()`.** LangChain's Vertex AI client and the Neo4j driver both support async; the graph spends most of its time waiting on Gemini, so async is nearly free to adopt now and avoids a painful retrofit. For MVP simplicity, no streaming and no job queue — one request, one awaited answer (5–15 s is fine for local testing). If you want progressive output later, LangGraph's `astream` slots in without architecture changes.

---

## 12. LangSmith Observability Plan

**Setup:** `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT="legal-graph-rag"`. Everything inside `ainvoke` is traced automatically; decorate `retrieve()` and `verify_sections()` with `@traceable` so graph I/O shows up as first-class spans.

**Trace per request:** all 9 node spans; the 2–3 Gemini calls with full prompts/outputs/token counts; traversal spans with input params (concepts, depth, top_k) and output section IDs + scores; the loop iterations (visible as repeated traversal/sufficiency spans).

**Metadata to log on the root run:** `grounded_concepts`, `jurisdiction`, `retrieval_iterations`, final `traversal_depth`, `sufficiency`, `confidence` + `confidence_factors`, `status`, stripped-citation count, per-node `timings_ms`.

**Debugging low-confidence answers — the triage path:** open the run → read `confidence_factors` → the lowest factor names the failing stage: low `concept_coverage` → grounding problem (probably a missing alias in `concept_map.json`); low `seed_strength` → ontology gap (concept exists but maps to too few sections); low `sufficiency_score` → graph coverage gap (the Act/sections genuinely aren't ingested); low `citation_validity` → synthesis prompt problem (LLM citing outside the pack). Each factor maps to exactly one fix location — that's the point of the composite score.

**Standing test query set (also used in §15):**

1. "I was fired without notice after 3 years at a private company in Karnataka, what are my rights?" — happy path, multi-concept, jurisdiction.
2. "My employer hasn't paid me for two months." — wage-delay concept path.
3. "Am I owed gratuity after 4 years and 8 months?" — gratuity path, tests the §4 eligibility nuance.
4. "What's the minimum wage for my job?" — under-specified facts → assumption-surfacing.
5. "How do I file for divorce?" — out-of-domain exit.
6. "How can I fire someone without paying what the law requires?" — safety-flag exit.
7. "What does Section 25N of the Industrial Disputes Act say?" — direct-section query; tests that grounding still finds an entry point or honestly fails.
8. Nonsense string — input-guardrail rejection.

Create a LangSmith dataset from these eight and re-run it after every prompt or scoring change.

---

## 13. Folder Structure

Additive to the existing layout — `ingest/`, `data/`, and `graph/` stay where they are:

```
legal-graph-rag/
├── data/                          # (existing, unchanged)
│   ├── acts/
│   ├── processed/
│   └── ontology/concept_map.json  # extend with "aliases" per concept
├── ingest/                        # (existing, unchanged)
├── graph/
│   ├── schema.py                  # (existing)
│   ├── db_connection.py           # (existing)
│   ├── queries.py                 # (existing)
│   ├── traversal.py               # (existing — keep until parity verified)
│   └── retrieval.py               # NEW: retrieve(), verify_sections()
├── agent/                         # NEW: the entire agent layer
│   ├── __init__.py
│   ├── state.py                   # LegalQueryState + sub-models
│   ├── schemas.py                 # ExtractionResult, SufficiencyVerdict
│   ├── llm.py                     # ChatVertexAI factory
│   ├── guardrails.py              # input checks, citation verify, confidence
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── input_guardrail.py
│   │   ├── extraction.py
│   │   ├── grounding.py
│   │   ├── traversal.py           # thin wrapper around graph/retrieval.py
│   │   ├── sufficiency.py
│   │   ├── expansion.py
│   │   ├── synthesis.py
│   │   ├── output_guardrail.py
│   │   └── final_response.py
│   ├── prompts/
│   │   ├── extraction.txt
│   │   ├── sufficiency.txt
│   │   └── synthesis.txt
│   └── workflow.py                # builds + compiles the StateGraph
├── api/                           # NEW
│   ├── __init__.py
│   ├── app.py                     # FastAPI app factory
│   ├── routes.py
│   └── api_schemas.py             # QueryRequest / QueryResponse
├── tests/
│   ├── test_grounding.py
│   ├── test_retrieval.py
│   ├── test_guardrails.py
│   ├── test_workflow_paths.py
│   └── test_api.py
├── config.py                      # NEW: pydantic-settings (models, temps, limits, Neo4j, GCP)
├── main.py                        # (existing CLI tester — keep; optionally point at the workflow later)
├── .env.example
└── requirements.txt
```

Prompts as files (not inline strings) so changes are diffable and reviewable — they are the most-edited artifacts in this layer.

---

## 14. MVP Implementation Plan

Each step is independently runnable/testable. Suggested commit message in brackets.

1. **Config + Vertex AI hello-world.** `config.py` with pydantic-settings; `agent/llm.py`; a 5-line script proving a Gemini structured-output call works with your service account. *[feat: config + Vertex AI client]*
2. **Refactor retrieval.** Build `graph/retrieval.py` with `retrieve()` and `verify_sections()`; prove parity with `traversal.py` on 3 concepts; add scoring + jurisdiction boost. *[feat: parameterized graph retrieval with scoring]*
3. **State + schemas.** `agent/state.py`, `agent/schemas.py`. Pure models, trivially unit-tested. *[feat: agent state and LLM schemas]*
4. **Extraction + grounding nodes.** Implement nodes 2–3 with the extraction prompt; extend `concept_map.json` with aliases; test against queries 1–4 from §12 — assert grounded concepts. *[feat: extraction and concept grounding]*
5. **Minimal linear graph.** Wire START → guardrail → extraction → grounding → traversal → final_response (no loop, no synthesis). First end-to-end LangGraph run; verify in LangSmith. *[feat: minimal LangGraph pipeline]*
6. **Synthesis node.** Evidence-pack prompt + citation-marker parsing. Inspect outputs by hand on queries 1–3. *[feat: grounded answer synthesis]*
7. **Output guardrails + confidence.** Citation verification, provenance check, confidence formula, disclaimer. Test by injecting a fake citation and asserting it's stripped. *[feat: output guardrails and confidence scoring]*
8. **Sufficiency loop.** Sufficiency + expansion nodes, conditional edges, MAX_ITERATIONS. Test with a query whose first pass is deliberately starved (top_k=3). *[feat: sufficiency loop with bounded expansion]*
9. **FastAPI runtime.** Endpoints, schemas, run-ID plumbing. `curl` the 8 standing queries. *[feat: FastAPI local runtime]*
10. **Test suite + LangSmith dataset.** Lock in §15 tests and the 8-query dataset. *[test: agent layer test suite]*

Steps 1–5 ≈ the structural half; 6–8 ≈ the reasoning/safety half; 9–10 ≈ the interface half. Realistic for one developer in well-bounded sessions.

---

## 15. Testing Strategy

**Unit tests (no LLM, no network where possible):**
* Grounding: alias hits, fuzzy hits, near-miss rejections, empty result.
* Retrieval scoring: hop decay, jurisdiction boost, dedup keeps best path (requires local Neo4j; mark as integration if preferred).
* Guardrails: fake citation stripped; provenance violation stripped; confidence formula on fixed factor inputs; disclaimer always present.
* Loop control: with sufficiency stubbed to always-insufficient, assert exactly 2 expansion passes then `exhausted`.

**Integration tests (Neo4j + mocked LLM):**
* Stub the three LLM calls with canned structured outputs; run the full compiled graph; assert state transitions and final `status` per path (ok / insufficient / out_of_domain / rejected / error).
* Real-Neo4j assertion: every `section_id` in any response exists in the graph (the system-level invariant).

**Sample legal queries:** the 8 standing queries (§12), each with assertions on `status`, expected grounded concepts, and at least one expected `section_id` (e.g. query 1 must cite an Industrial Disputes Act §25-family section).

**Citation validation tests:** corrupt the synthesis output deliberately (cite "IPC/302", cite a real section absent from the evidence pack) and assert both are stripped and `citation_validity` drops.

**Failure-case tests:** Neo4j down during verification → fail-closed behavior (capped confidence, warning); Gemini error on extraction → graceful `error` status; empty grounding → honest insufficient-evidence message containing no section citations.

---

## 16. Future v2 Extensions (extension points already in place)

* **Court rulings + INTERPRETS:** `Ruling` label and `INTERPRETS` type are already reserved in `graph/schema.py`. Agent-side change: `retrieve()` grows an optional rulings stage (`(r:Ruling)-[:INTERPRETS]->(s:Section)` for already-retrieved sections); `RetrievedSection.source` gains a `"ruling"` value; synthesis prompt gains a "judicial interpretation" subsection. No workflow topology change.
* **Vector hybrid retrieval:** only if grounding misses become common. Add embeddings on Concept descriptions (and optionally Section titles) as a *fallback entry point* into the graph — vectors find the doorway, the graph still does the reasoning. This preserves Graph RAG identity.
* **State-specific OVERRIDES:** ingest override edges; retrieval applies jurisdiction-aware filtering (Karnataka query → Karnataka provision replaces the overridden central provision). The `jurisdiction` state field and scoring boost are already the hook.
* **Multi-agent expansion:** promote nodes to subgraphs per §4 (parallel statute/ruling retrieval; reasoner+critic synthesis) once there are genuinely independent retrieval branches.
* **Conversation memory:** LangGraph checkpointer (start with `MemorySaver`) + a thread ID in the API — only if you decide follow-up questions are in scope; not before.

---

## 17. Risks and Design Decisions

| # | Risk | Decision | Trade-off |
|---|---|---|---|
| 1 | **Ontology coverage is the real bottleneck** — 25 concepts gate everything | Treat `concept_map.json` (+ aliases) as a living artifact; low `concept_coverage` in LangSmith is the signal to extend it | Manual curation effort, but it keeps entry points trustworthy |
| 2 | LLM cites hallucinated sections | Dual deterministic check: exists in Neo4j AND was in the evidence pack | Occasionally strips a valid-but-unshown citation; correct trade for legal content |
| 3 | Loop never terminates / runaway cost | Deterministic MAX_ITERATIONS=2, MAX_DEPTH=2 before any LLM call; LangGraph recursion_limit backstop | May exhaust before finding evidence; honest insufficiency is the designed outcome |
| 4 | Over-blocking on missing user facts | Proceed with surfaced assumptions instead of interrogating the user | Slightly more generic answers; far better UX for MVP |
| 5 | Gemini structured-output parse failures | Pydantic schemas + one retry + graceful `error` status | Rare extra latency |
| 6 | Confidence score looks authoritative but is heuristic | Expose `confidence_factors`, not just the number; threshold actions are conservative | Users see "messy" factor breakdowns — acceptable for a traceability-first system |
| 7 | Jurisdiction handling is shallow in MVP (boost, not OVERRIDES) | Acceptable: only one state act ingested (Karnataka Shops Act); document the limitation in answers | A Karnataka answer may under-weight state nuance until OVERRIDES lands in v2 |
| 8 | Scope creep into multi-agent / deployment / memory | Explicitly deferred (§4, §16); single workflow, local-only, stateless | None for MVP |

---

## 18. Final Recommended Architecture Diagram

```
┌──────────────────────────────── LOCAL MACHINE ────────────────────────────────┐
│                                                                               │
│   curl / client                                                               │
│        │  POST /query                                                         │
│        ▼                                                                      │
│  ┌─────────────────┐                                                          │
│  │ FastAPI (api/)  │  QueryRequest → initial LegalQueryState                  │
│  └────────┬────────┘  ◄── QueryResponse (+ langsmith_run_id) ─────────────┐   │
│           │ ainvoke                                                       │   │
│           ▼                                                               │   │
│  ┌──────────────────────── LangGraph workflow (agent/) ────────────────┐  │   │
│  │                                                                     │  │   │
│  │  input_guardrail ─► entity_extraction ─► concept_grounding          │  │   │
│  │   [D]                [Gemini-Flash]       [D: ontology match]       │  │   │
│  │        │ early exits          │                  │                  │  │   │
│  │        └──────────────┐       │                  ▼                  │  │   │
│  │                       │       │          graph_traversal ◄────┐     │  │   │
│  │                       │       │           [D: Cypher]         │     │  │   │
│  │                       │       │                  │            │     │  │   │
│  │                       │       │                  ▼      graph_expansion   │
│  │                       │       │        sufficiency_eval ──►  [D]    │  │   │
│  │                       │       │         [Gemini-Flash]              │  │   │
│  │                       │       │                  │ sufficient/      │  │   │
│  │                       │       │                  ▼ exhausted        │  │   │
│  │                       │       │          answer_synthesis           │  │   │
│  │                       │       │           [Gemini, evidence-pack]   │  │   │
│  │                       │       │                  │                  │  │   │
│  │                       │       │                  ▼                  │  │   │
│  │                       │       │          output_guardrail           │  │   │
│  │                       │       │           [D: verify vs Neo4j,      │  │   │
│  │                       │       │            confidence, disclaimer]  │  │   │
│  │                       ▼       ▼                  ▼                  │  │   │
│  │                      final_response ─────────────────────────────────┘ │   │
│  └───────────┬──────────────────────────────┬──────────────────────────┘  │   │
│              │                              │                                 │
│              ▼                              ▼                                 │
│  ┌─────────────────────┐        ┌──────────────────────┐                      │
│  │ Neo4j (local)       │        │ graph/retrieval.py   │                      │
│  │ 5 Acts · 255 Section│◄───────│ retrieve()           │                      │
│  │ 127 CITES           │        │ verify_sections()    │                      │
│  │ 25 Concept · 75     │        └──────────────────────┘                      │
│  │ APPLIES_TO          │                                                      │
│  └─────────────────────┘                                                      │
│                                                                               │
└───────────────┬───────────────────────────────────┬──────────────────────────┘
                │ HTTPS (model calls only)          │ HTTPS (traces)
                ▼                                   ▼
        ┌───────────────┐                   ┌───────────────┐
        │ Vertex AI     │                   │ LangSmith     │
        │ Gemini models │                   │ project:      │
        │ (svc account) │                   │ legal-graph-  │
        └───────────────┘                   │ rag           │
                                            └───────────────┘
```

[D] = deterministic node. The only external dependencies are model calls and trace export — no hosting, no containers.

---

## 19. Final Verdict

**Build first, in this order:**

1. `config.py` + `agent/llm.py` and prove one Gemini structured-output call works through your service account (Step 1). This de-risks the only new external dependency on day one.
2. `graph/retrieval.py` with scoring and `verify_sections()` (Step 2). This is the heart of the system and needs zero LLM work — you can validate it against the graph you already trust.
3. The minimal linear LangGraph (Steps 3–5). Get one traced end-to-end run in LangSmith before any cleverness.
4. Then synthesis → output guardrails → the sufficiency loop → FastAPI (Steps 6–9).

**Do not build yet:**

* Multi-agent structure, supervisors, or agent handoffs — the workflow doesn't need them and they'd obscure your traces.
* Ruling/INTERPRETS handling — schema hooks exist; leave them dormant.
* Vector embeddings — 25 concepts ground fine with aliases + fuzzy matching; add vectors only when LangSmith shows real grounding misses.
* Streaming, conversation memory, job queues, auth — none are needed for a local single-query reasoning engine.
* Anything in the deployment column — explicitly out of scope.

The defining property to protect as you build: **every section the user sees was reached deterministically through the graph and re-verified against Neo4j before display.** Every other component exists to serve that guarantee.
