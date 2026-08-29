# CLAUDE.md — Legal Graph RAG

Instructions for Claude Code working in this repository. Read this fully before making changes.

---

## 1. What this project is

A **Graph RAG legal reasoning engine** for Indian employment law. A user asks a plain-language legal question; the system grounds it to known legal concepts, deterministically traverses a Neo4j knowledge graph to retrieve relevant statutory sections, and uses an LLM only to synthesize a grounded answer from the retrieved text. Every section the user sees was reached deterministically through the graph and verified against Neo4j before display. That guarantee is the identity of the project — protect it above all else.

This is a **single LangGraph state-machine workflow** (NOT multi-agent). It is a pipeline with one conditional loop, not a negotiation between autonomous agents.

---

## 2. Stack

- **Python 3.10**
- **LangGraph** — orchestration (the compiled `StateGraph` is the single source of control flow)
- **LangChain** — model/tool layer
- **Gemini 2.5 Flash via Vertex AI** — the LLM (`langchain-google-vertexai`, `ChatVertexAI`)
- **Neo4j** — graph database (READ-ONLY at runtime)
- **FastAPI** — local runtime (`api/` package — **complete**)
- **LangSmith** — observability (ambient tracing of the LangGraph run + all 3 LLM calls). Requires THREE `.env` vars: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=<key>`, `LANGCHAIN_PROJECT=CourseLanggraph`. Two gotchas, both load-bearing:
  - **The enable flag is matched case-sensitively against the literal `"true"`** (`langsmith.utils.tracing_is_enabled` → `var == "true"`). `True`/`TRUE` silently disable tracing with no error.
  - **`load_dotenv()` must run before any LangChain/LangGraph/Vertex import** so the vars are in `os.environ` when the run is invoked. `graph_agent.py` already does this at the top (lines 33–35); preserve that ordering. `LANGCHAIN_PROJECT` is the project **name** (LangSmith resolves it) — a wrong/nonexistent name routes traces away from the dashboard rather than erroring.

Auth for Vertex AI is via **Application Default Credentials (ADC)** — `gcloud auth application-default login`. Do NOT set `GOOGLE_APPLICATION_CREDENTIALS`. Required env vars: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.

---

## 3. Non-negotiable rules

These encode the project's core design. Violating them defeats the purpose of the system.

1. **The LLM is called in exactly THREE places**: entity extraction, sufficiency evaluation, and answer synthesis. Never add an LLM call anywhere else. Grounding, traversal, guardrails, confidence scoring, and disclaimers are all deterministic Python/Cypher.
2. **The agent layer is READ-ONLY against Neo4j.** Never write Cypher that mutates the graph (no MERGE/CREATE/SET/DELETE in the agent or graph read paths).
3. **The LLM never invents a section.** It only describes sections the graph returned. Synthesis cites only `section_id`s present in the evidence pack.
4. **Loop termination is never decided by the LLM.** Hard limits (`MAX_ITERATIONS`, `MAX_HOPS`) are enforced in deterministic code BEFORE any LLM call.
5. **Every citation must be verified against Neo4j** before reaching the user, AND must have been in the retrieval set (the LLM may only cite what it was shown).
6. **No raw dictionaries between agents.** Use the typed contracts in `agents/state.py` and `agents/graph_state.py`.
7. **Prompts live as files** in `agents/prompts/`, never as inline strings. The synthesis system prompt is **composed** from `synthesis_base.txt` (the shared, integrity-critical rules — kept in ONE place so they can never drift) plus a persona overlay (`synthesis_citizen.txt` or `synthesis_lawyer.txt`); both halves are files.
8. **Repealed law is never cited, and state law is never cited outside its
   state.** `data/ontology/act_metadata.json` is the single source of both
   facts; `ingest/act_metadata_loader.py` mirrors them onto the Act nodes and
   `graph/queries.py` joins them onto every section it returns. Filtering
   happens in `graph/traversal.py` **before** confidence scoring and before BFS
   expansion — filtering later would let a repealed section act as a primary
   anchor and steer which live sections get retrieved. An Act is only marked
   repealed when the repealing Act is in the corpus AND the repeal is stated in
   retrieved text (`repeal_authority` must be a real `section_id`).
9. **The citation-marker format lives in exactly one place: `agents/citations.py`.**
   Both the parser (synthesis) and the stripper (final response) import it.
   Two copies of that regex is one too many — a marker the parser fails to
   recognise is never verified, and if the stripper misses it too it reaches
   the user unchecked. This is not hypothetical, and it has now happened
   **twice**: sub-section markers (`[IDA_1947_S7(1)]`) matched neither of the
   two old copies, and then GROUPED markers
   (`[IRC_2020_S43, IRC_2020_S44, IRC_2020_S49]`) — which the lawyer persona
   emits constantly — matched the single-id pattern that replaced them, because
   it required `]` straight after the first id. Both leaked unverified ids into
   answers while `verified_section_ids` listed none of them. A grouped marker is
   now parsed id-by-id and re-emitted containing only the parts that survived
   verification. Markers are verified on their BASE section id.
   `tests/test_units.py` pins all of this; run it after ANY change here.
10. **Persona tailors presentation only, never substance.** The selected persona changes the synthesis tone/technicality/structure, how citation markers are *rendered*, and the final-response trailer — it NEVER changes what is retrieved, verified, or allowed to be cited, and it does NOT add an LLM call (rule 1 still holds: exactly three). Removing a *verified* marker from the citizen's visible text is presentation: verification already happened in `output_guardrail_node`, which runs first. Removing an *unverified* marker is rule 5 and is persona-independent.

---

## 4. Folder structure

```
legal-graph-rag/
├── graph_agent.py          # ENTRY POINT (CLI) — builds & compiles the LangGraph workflow
├── config.py               # pydantic-settings — SINGLE source of all runtime config
├── main.py                 # legacy CLI traversal tester (no LLM/agents) — keep
├── requirements.txt
├── .env                    # Neo4j creds + GCP project/location + LangSmith keys
│
├── api/                    # FastAPI HTTP layer — COMPLETE
│   ├── __init__.py         # empty
│   ├── app.py              # FastAPI app factory — `uvicorn api.app:app --reload --port 8000`
│   ├── models.py           # QueryRequest / ConfidenceFactors / QueryResponse
│   └── routes/
│       ├── __init__.py     # empty
│       ├── query.py        # POST /query
│       ├── health.py       # GET  /health
│       └── info.py         # GET  /graph/stats  and  GET  /concepts
│
├── agents/                 # the agent layer
│   ├── graph_state.py      # LegalQueryState — LangGraph orchestration state (Pydantic)
│   ├── state.py            # domain dataclasses (RetrievalResult, SectionContext, etc.) — zero external deps
│   ├── schemas.py          # Pydantic schemas for structured LLM output
│   ├── llm.py              # ChatVertexAI factory (reads model/GCP from config)
│   ├── ontology.py         # deterministic concept grounding (loads concept_map.json)
│   ├── persona.py          # persona vocabulary ("citizen" | "lawyer") + normalize/match helpers
│   ├── nodes/              # one file per LangGraph node
│   │   ├── extraction_node.py        # LLM call 1
│   │   ├── grounding_node.py         # deterministic
│   │   ├── retrieval_node.py         # deterministic (wraps graph/traversal.py)
│   │   ├── sufficiency_node.py       # LLM call 2
│   │   ├── expansion_node.py         # deterministic
│   │   ├── synthesis_node.py         # LLM call 3
│   │   ├── output_guardrail_node.py  # deterministic — citation verify + confidence + disclaimer
│   │   └── final_response_node.py    # deterministic — assembles final_answer for every exit path
│   ├── citations.py        # SINGLE source of the [SECTION_ID] marker format —
│   │                       #   parsing AND stripping (never duplicate this regex)
│   └── prompts/            # extraction.txt, sufficiency.txt,
│                           #   synthesis_base.txt (shared integrity rules) +
│                           #   synthesis_citizen.txt / synthesis_lawyer.txt (persona overlays)
│
├── graph/                  # Neo4j access — DO NOT add LLM logic here
│   ├── schema.py           # node/relationship definitions (single source of truth)
│   ├── db_connection.py    # Neo4j driver (reads creds from config)
│   ├── queries.py          # all Cypher (the only place raw Cypher lives)
│   ├── traversal.py        # deterministic BFS traversal + temporal/territorial filters
│   ├── act_registry.py     # offline reader of act_metadata.json — in-force status,
│   │                       #   jurisdiction resolution, explicit section-ref parsing
│   └── ranking.py          # deterministic relevance scoring (BM25 + graph signals)
│
├── tools/                  # BUILD-TIME only, never imported at runtime
│   ├── build_concept_map.py    # rebuilds + validates data/ontology/concept_map.json
│   ├── section_concept_map.json  # curated section -> concept table (the source)
│   └── new_concepts.json         # concept definitions added on top of the original 25
│
├── tests/
│   ├── golden_queries.json # 50 queries with expectations
│   └── verify.py           # end-to-end harness, checks answers vs the raw corpus
│
├── ingest/                 # COMPLETE — DO NOT MODIFY
│   ├── pdf_parser.py
│   ├── graph_builder.py
│   └── ontology_loader.py
│
└── data/
    ├── acts/               # source PDFs
    ├── ontology/
    │   ├── concept_map.json    # 54 concepts + 359 aliases — the grounding dictionary.
    │   │                       #   GENERATED by tools/build_concept_map.py; edit the
    │   │                       #   sources in tools/, not this file.
    │   └── act_metadata.json   # per-Act jurisdiction + in-force/repeal status —
    │                           #   the single source of truth for both filters
    └── processed/          # sections.jsonl, relationships.jsonl
```

---

## 5. Current status

**The pipeline is COMPLETE end-to-end.** A query now flows from raw text all the
way to a verified, confidence-scored, disclaimer-bearing final answer:

```
extraction → grounding → traversal → sufficiency → [conditional]
     "expand" → expansion → traversal            (loop, ≤ MAX_ITERATIONS)
     "end"    → synthesis → output_guardrail → final_response → END
```

**Complete and working:**

- Ingestion layer (`ingest/`) — graph is populated: **7 Acts, 523 Sections, 344 CITES,
  54 Concepts, 466 APPLIES_TO, 3 OVERRIDES**. Concept coverage is
  **282/383 in-force sections (74%)**. The uncovered sections are pure
  machinery (power to make rules, delegation, protection of action taken in
  good faith, constitution of boards, accounts/audit/budget) and PDF artifacts,
  which carry no answerable rule.
  Rebuild with `python -m tools.build_concept_map` then
  `python -m ingest.ontology_loader`.

  **The four Labour Codes.** Three of the four are now ingested — Code on Wages
  2019, Industrial Relations Code 2020, Code on Social Security 2020 — all
  brought into force on **21 November 2025** (S.O. 5320(E); the commencement
  footnote is printed in the IRC text itself). Between them they repealed the
  MWA 1948, IDA 1947 and PGA 1972, which is why all three are suppressed. The
  fourth, the **OSH Code 2020**, is NOT ingested; it subsumes the central
  factories/shops regime, so the Karnataka Shops Act remains the only source of
  hours/leave/weekly-holiday answers and stays Karnataka-only.

  Ingesting SSC 2020 also CLOSED coverage gaps that used to be honest refusals:
  provident fund, ESI, maternity benefit, injury compensation, and gig/platform
  and unorganised workers are all answerable now. Nine new concepts cover them.

  *"Labour Code 2026" is not a separate statute.* Act No. 1 of 2026 is an
  amending Act (it substituted IRC s.104(1) w.e.f. 21-11-2025); the ingested
  PDFs already incorporate it.
- **Temporal + territorial layer** — `data/ontology/act_metadata.json` +
  `ingest/act_metadata_loader.py`. THREE Acts are marked repealed and suppressed:
  MWA 1948 (by COW 2019 s.69), IDA 1947 (by IRC 2020 s.104), PGA 1972 (by SSC 2020
  s.164) — all w.e.f. 21 Nov 2025. KSEA 1961 is Karnataka-only and is withheld
  from users in other states. Every suppression is reported, never silent.
- **Ranking** (`graph/ranking.py`) — BM25 over the candidate set plus relevance,
  hop distance, concept-hit count and act priority. The evidence pack is ordered
  and numbered, and the synthesis prompt tells the model to work down from #1.
- **Verification harness** (`tests/`) — 55 golden queries run end-to-end against
  real Neo4j + Gemini, checked against the raw corpus.
- Graph layer (`graph/`) — schema, queries, deterministic traversal.
- Full agent pipeline, all three LLM calls + both deterministic guardrail nodes:
  - `synthesis_node` (LLM call 3) — verified working; cites only evidence-pack section_ids, short-circuits the empty-retrieval path with no LLM call. **Persona-aware:** the system prompt is `synthesis_base.txt` + the selected persona overlay (citizen = plain-language/reassuring; lawyer = technical/section-by-section).
  - `output_guardrail_node` — verifies every citation against BOTH the retrieval set (provenance) AND Neo4j (existence), strips failures with a warning, computes the deterministic confidence score (§6), downgrades to `insufficient_evidence` below `MIN_CONFIDENCE`, and injects the disclaimer in code. Degrades honestly if Neo4j is unreachable (provenance-only + warning).
  - `final_response_node` — pure formatting (NO LLM, NO Neo4j), assembles `final_answer` for all five terminal statuses with safety precedence (see §6); strips unverified inline `[SECTION_ID]` markers; wrapped so it never raises. **Persona-aware trailer:** lawyer gets "Verified citations:" (raw ids) + the numeric confidence factor breakdown; citizen gets "The law behind this answer: …" in readable form ("Section 18 of the Code on Wages") and a worded confidence band (high/moderate/low) with no jargon, capped at "moderate" when the evidence is partial.
- `config.py` — pydantic-settings is the single source of all runtime config (model, per-call temps/token budgets, all hard limits, confidence weights, Neo4j creds, GCP project/location). Every former hardcoded constant now reads from here; override any value via an env var of the same name.

**Verified terminal statuses:** `ok`, `insufficient_evidence`, `out_of_domain`, `rejected`, `error` — all exercised (live + offline) and producing correct output.

**Not built yet (the roadmap):**

- Unit tests — **started**. `tests/test_units.py` is a fast offline layer (no
  Neo4j, no Vertex, runs in ~1s) covering `agents/citations.py`, grounding, the
  companion layer, and the `act_registry` temporal helpers. Run it with
  `python -m tests.test_units`. Still uncovered: the confidence formula and the
  node contracts — extend this file rather than starting another.
- Case law. `Ruling` / `INTERPRETS` remain defined with no data.
- Coverage gaps the temporal model now makes visible: `payment_of_wages_act_1936.pdf`
  sits in `data/acts/` un-ingested, and the Payment of Bonus Act 1965, Equal
  Remuneration Act 1976, Industrial Relations Code 2020 and Social Security Code
  2020 are absent. Ingest a repealing Code and its repeal wires up with no code
  change (see the repeal rule in `act_metadata.json`).
- `graph/retrieval.py` — optional refactor of `traversal.py` into parameterized `retrieve()` + `verify_sections()`.

---

## 6. Key architectural facts

- **State:** `LegalQueryState` (Pydantic) flows through every node; each node returns a partial dict update. Step-7 fields: `verified_section_ids`, `confidence`, `confidence_factors`, `status`, `disclaimer`, `final_answer`. Also carries `persona` (selected at login; empty normalizes to `"citizen"`).
- **Persona-aware output:** the persona (canonical `"citizen"` | `"lawyer"`, resolved by `agents/persona.py`; Lawyer/Judge/Advocate all map to `"lawyer"`) is set on the initial state. `run(raw_query, persona=None)` threads it in (default from `cfg.DEFAULT_PERSONA`); the CLI prompts for it once at login. It is consumed only by `synthesis_node` (prompt selection) and `final_response_node` (marker rendering + trailer). All integrity guarantees hold identically for both personas.
- **Citizen readability rules (presentation only).** `synthesis_citizen.txt` enforces a fixed shape — a direct answer in the opening 2-3 sentences, then `What the law says` / `Where you stand` / `What you can do next`, plus `What this answer doesn't cover` only when `sufficient=False` — under hard budgets (200-350 words, ≤3 sections, no verbatim statutory quoting, never open a paragraph with a section number, never assume the events happened to the reader). `final_response_node` then: removes the verified `[SECTION_ID]` markers from the visible text (a layperson cannot use a machine id; the law is already named in words), renders the trailer as `Section 18 of the Code on Wages` using `act_name`/`section_number` from `state.retrieval`, and caps the worded confidence at "moderate" whenever the answer admits a gap — note this keys off the *sufficiency verdict*, not `status`, because the loop can exhaust with `sufficient=False` and still score above `MIN_CONFIDENCE` (status stays `ok`). The numeric score is never altered. Lawyer output is untouched by all of this.
- **`act_name` is joined in the graph layer, not recalled by the LLM.** Section nodes store only `act_id`; `graph/queries.py` joins `(:Act)-[:HAS_SECTION]->(s)` in every read path (`get_sections_for_concept`, `get_sections_for_exact_concept`, `get_neighbors`, `get_subgraph`) via `_with_act_name`. Before this, the evidence pack showed an empty act name and the model supplied act names from its own memory.
- **All tunable constants live in `config.py`** (`from config import cfg`). Do NOT reintroduce hardcoded models/temps/limits in node files — add a field to `config.py` instead.
- **Confidence formula (deterministic, implemented in `output_guardrail_node`):**
  `confidence = 0.35·concept_coverage + 0.25·seed_strength + 0.20·sufficiency_score + 0.20·citation_validity`
  (weights are `cfg.CONFIDENCE_W_*`). Each factor ∈ [0, 1]:
  - `concept_coverage` — fraction of EXTRACTED PHRASES that grounded to at least
    one concept, computed from `state.ungrounded_phrases` (1.0 if no extraction
    but something grounded). NOT `len(grounded)/len(extracted)`: grounding
    unions its matches, so that ratio could exceed 1.0 while hiding a phrase
    that grounded to nothing, and scored 0.5 when two phrases correctly
    collapsed onto one concept.
  - `seed_strength` — reuses traversal confidence (1.0 primary anchors, 0.6 supporting-only, 0.0 none).
  - `sufficiency_score` — 1.0 sufficient · 0.5 insufficient-but-sections-retrieved · 0.0 none.
  - `citation_validity` — verified ÷ cited citations.
  Below `cfg.MIN_CONFIDENCE` (0.4) the guardrail sets `status = "insufficient_evidence"`.
- **Hard limits (all in config):** `MAX_RETRIEVAL_ITERATIONS = 2`; `MAX_HOPS_DEFAULT = 2`, capped at `MAX_HOPS_CAP = 4`; LangGraph `recursion_limit` (`LANGGRAPH_RECURSION_LIMIT`, ~25) passed to `invoke()` as a backstop.
- **Terminal status & safety precedence:** `final_response_node` resolves the final status as
  `error > rejected (safety_flag) > out_of_domain (¬in_domain) > guardrail status (ok | insufficient_evidence)`.
  For `rejected` / `out_of_domain` / `error` it shows a fixed honest message and NO legal content — even if synthesis ran upstream.
- **Known gap (intentional, documented):** there is NO early-exit routing yet. Every path flows through `synthesis → output_guardrail → final_response`. Synthesis self-short-circuits (no LLM call) when retrieval is empty — which covers most out-of-domain queries — but a `safety_flag` query that still grounds to real concepts WILL spend a synthesis LLM call whose output `final_response` then discards. Adding a conditional early-exit edge after extraction is a future optimization, not a correctness bug (the safety guarantee holds via final-response precedence).
- **Exhausted ≠ error:** when the loop exhausts, synthesis still runs but produces an honest partial answer.
- **Retrieval order is meaningful and the model is told so.** `graph/ranking.py`
  scores the union across all grounded concepts against the user's query; the
  evidence pack is emitted numbered (`#1`, `#2`, …) and the synthesis prompt
  instructs the model to work down from the top. Ranking that the model cannot
  see does nothing — on "fired without notice after 3 years" the model cited a
  lower-ranked provision and ignored the #1 operative section until the pack was
  numbered.
- **Every answer states WHEN the cited law came into force.** Deterministic, in
  `final_response_node` via `act_registry.commencement_notes()` — never left to
  the LLM. The three Codes commenced **21 November 2025**, so conduct before
  that date is still governed by the Acts the engine suppresses; an answer that
  gives the current position without dating it lets a practitioner apply the IRC
  to a pre-commencement dispute. The citizen also gets `replacement_note()`
  ("the Industrial Disputes Act was replaced by … on 21 November 2025"), because
  a correct answer that contradicts everything published before that date reads
  as wrong unless it says why. That notice is filtered to Acts whose **successor
  was actually cited** — `suppressed_acts` accumulates across every concept
  traversed, so unfiltered it told a wages query about the Gratuity Act.
- **Deterministic entitlement calculator** (`agents/calculators.py`) — turns "you
  get fifteen days' pay per year" into an actual rupee figure, computed in pure
  Python, never by the LLM (rule 1 is unaffected — still exactly three LLM
  calls). Extraction now also captures `monthly_salary` when the user states
  one. Gated on PROVENANCE like any citation: a calculation is only offered
  when its section was actually retrieved for this query, and only reaches the
  user after `output_guardrail_node` re-verifies it exists in Neo4j
  (`verified_calculations` — the calculator gets no exemption from rule 5 for
  being arithmetic instead of prose). Synthesis is told a verified figure will
  be appended and to describe the entitlement in WORDS ONLY, never restate its
  own total — two numbers that could disagree is the one failure mode this
  design cannot allow. Gratuity (`SSC_2020_S53` + Explanation 3) and notice pay
  (`IRC_2020_S70`/`S79`, a flat multiply) compute a real figure because the
  retrieved text gives the COMPLETE formula, divisor included. Retrenchment
  COMPENSATION ("fifteen days' average pay" per year) deliberately never
  returns a number: "average pay" is defined (`IRC_2020_S2(d)`) as a monthly
  average with no stated day-count conversion, and reusing gratuity's
  divide-by-26 there would be inventing a rule the text does not contain — the
  same fabrication rule 3 exists to prevent, just caught in Python instead of a
  prompt. Gratuity eligibility is similarly conservative: below 5 years is
  refused outright rather than assumed, even though `SSC_2020_S54`'s 240-day
  continuous-service deeming rule could genuinely bring a 4-year-8-month tenure
  up to 5 — the calculator has no way to check that from `years_of_service`
  alone, so it says so instead of guessing in either direction.
- **Pre-commencement (temporal mismatch) check** — the ONE failure mode more
  dangerous than an honest gap: a confidently WRONG answer. "I was retrenched
  in August 2025" grounds correctly and retrieves real, verified IRC 2020
  sections — but the IRC did not exist in August 2025 (commenced 21 Nov 2025).
  Citation verification does not catch this: it checks a section EXISTS and was
  SHOWN to the model, not that it APPLIED on the date in question. Extraction
  now captures `event_date` (normalized `YYYY-MM-DD`/`YYYY-MM`/`YYYY`; today's
  date is passed alongside the query so relative phrases like "last August"
  resolve to the right year); `retrieval_node` compares it against the
  commencement date of every PRIMARY act via `act_registry.commencement_conflicts()`
  and stores `state.temporal_conflicts`. This is enforced in THREE places, not
  one, following the project's ask-vs-enforce split: synthesis is told to
  address it first in its own opening; `output_guardrail_node` caps confidence
  at `cfg.TEMPORAL_MISMATCH_CONFIDENCE_CAP` (0.5); `final_response_node`
  PREPENDS a deterministic banner before the model's own text regardless of
  whether the model complied — a warning placed after a confident paragraph is
  a warning most readers have stopped reading for. Scoped to PRIMARY acts only,
  never companion/remedy-only ones, since those are the substantive law the
  answer actually asserts applies.
- **Companion (remedy) concepts** — `agents/ontology.COMPANION_CONCEPTS`, a
  curated concept→concept adjacency, deterministic, no LLM. A user asks what
  their position is and never thinks to ask which forum hears it, so retrieval
  returned the substantive law and no remedy; the citizen prompt correctly
  refuses to invent a forum or a deadline, so "What you can do next" degraded to
  "ask your employer" and "check your contract". Grounding to a grievance
  concept now also retrieves its remedy family. Three rules keep them subordinate:
  they are stored in `state.companion_concepts` (NOT merged into
  `grounded_concepts`, which would inflate `concept_coverage` with law the user
  never asked for); `relevance_for()` never lets them contribute a PRIMARY; and
  they never contribute to `seed_strength`. Their share of the pack is both
  floored AND capped at `cfg.COMPANION_SECTION_SLOTS` — uncapped they took 13 of
  15 slots on a wage query and crowded out the wage provisions.
  They are tagged **`(REMEDY)`** in the evidence pack (`SectionContext.via_remedy`,
  set in retrieval, rendered by `synthesis_node._format_section`) instead of
  plain `(SUPPORTING)`. Retrieval knows for certain which sections carry the
  forum; asking the prompt to infer it from section titles did not work — the
  remedy sections sat at #9/#10 and were ignored while "What you can do next"
  fell back to "review your contract". Both persona prompts now key off the tag.
- **A grounding alias must not reduce to a generic English word.** Stage-2
  overlap works on content tokens, so the alias "what happens to the employer"
  (of `penalties for employer offences`) reduces to `happen` — rare across the
  vocabulary, therefore *distinctive*, therefore a perfect-scoring single-token
  match. Every "what happens to…" question grounded to employer penalties; on
  the transfer-of-undertaking query it pulled six penalty and inspection
  sections in as PRIMARIES and buried the sections that answered it. Such words
  belong in `_GROUNDING_STOPWORDS` (`happen`, `let`, `lost` are there now).
  This costs nothing: stage-1 substring matching is not stopworded, so the
  verbatim alias still grounds. When adding aliases, check what they reduce to.
- **Grounding is a three-stage cascade** (`agents/ontology.py`), all deterministic:
  (1) substring match on concept names/aliases, (2) **token-overlap paraphrase
  matching**, unioned with stage 1 rather than used as a fallback, and (3) fuzzy
  n-gram matching for typos. Tokens are stemmed (`_norm_token`) for plurals AND
  the verb/noun split, because the vocabulary is written in nouns while users
  write verbs: "must an employer RETRENCH workmen?" shares no raw token with
  `retrenchment`, so it used to ground only to `compensation for injury at work`
  on the stray word "workmen". The single-token distinctiveness guard is waived
  when the token covers a concept's whole CANONICAL NAME — for a one-word
  concept the user has written the concept itself, yet document frequency counts
  it as common *because* it is the topic. Keyed on the name, never an alias:
  exempting aliases would let "employ" (DF 9) ground every query containing the
  word "employment". Stage 2 is what bridges the extraction LLM's own
  wording to the curated vocabulary — it writes "appeal", not the alias "appeal
  against an order". Stage 1 must NOT short-circuit stage 2: "punishment for
  non-payment of wages" contains the alias "payment of wages", so short-circuiting
  grounded a penalties question to `salary delay`. Grounding deliberately favours
  recall over precision, because ranking and the cap clean up over-matching while
  under-matching costs the user an answer outright; the `in_domain` gate, not
  grounding, is what keeps out-of-domain queries out.
- **Explicit section references bypass grounding.** "What does Section 25N say?"
  is a lookup, not a situation: it grounds to `industrial dispute` and never
  returns 25N. `retrieval_node._direct_section_lookups` resolves such references
  by id (`graph.act_registry.resolve_section_references`), verifies them against
  Neo4j, applies the same temporal/territorial filters, and injects them as hop-0
  primaries.

---

## 7. Validation-failure philosophy

Degrade honestly rather than block. A stripped-citation answer with a warning and lower confidence beats a refusal — EXCEPT for safety flags and out-of-domain queries, which exit before any legal content is generated. Never block on missing user facts (e.g. no tenure/state); proceed and surface assumptions as warnings.

---

## 8. How to run / test

**Activate the venv first.** Every command below assumes it. The project's
dependencies live in `venv/`, NOT in the system Python — the machine-wide
interpreter has pydantic 1.x and none of neo4j/langgraph/langchain-google-vertexai,
so running anything without activating gives
`ModuleNotFoundError: No module named 'pydantic_settings'` from `config.py`.
That is an activation problem, not a missing install; `pip install` into the
global interpreter is the wrong fix.

```powershell
.\venv\Scripts\Activate.ps1
```

```powershell
# CLI traversal tester (no LLM)
python main.py

# Agent workflow end-to-end (interactive — first prompts for a persona/login
#   [1] Normal Citizen / [2] Lawyer-Judge-Advocate, then "Enter your legal query:")
python graph_agent.py

# FastAPI server — interactive docs at http://localhost:8000/docs
uvicorn api.app:app --reload --port 8000

# Offline grounding check (no Neo4j/network)
python -m agents.ontology

# Vertex AI auth smoke test
python -m agents.llm

# Verify LangSmith tracing (non-interactive single query, then check the dashboard)
"2`nwhat does Section 25N of the Industrial Disputes Act say?`nexit" | python graph_agent.py
```

**Verify tracing:** after running any query above, a new trace must appear in the
**CourseLanggraph** project:
https://smith.langchain.com/o/e5f13691-d857-54eb-90ec-878eae6bc782/projects/p/027975bd-c093-472b-9753-5691347082cf
In the UI, confirm: a single root run named `LangGraph`; the 8 pipeline nodes as
child runs (`entity_extraction`, `concept_grounding`, `graph_traversal`,
`sufficiency_evaluator`, `graph_expansion`*, `answer_synthesis`,
`output_guardrail`, `final_response`); exactly **3** `ChatVertexAI` LLM child
runs with non-zero token counts. Zero traces almost always means the `.env`
enable flag is not the literal lowercase `true` (§2). (*`graph_expansion` only
appears when the sufficiency loop runs.)

**Verification harness — run this after ANY change to retrieval, grounding,
prompts or the ontology:**

```powershell
python -m tests.test_units               # fast offline unit checks (~1s) — run FIRST
python -m tests.verify                  # all 50 golden queries (~11 min)
python -m tests.verify --only T04 T09    # a subset, by id
python -m tests.verify --verbose         # print every final answer
python -m tests.verify --workers 6       # more parallelism (watch Vertex 429s)
```

It checks universal invariants on every query (citations real + in the retrieval
set, no repealed law, no out-of-state law, no unverified marker in the visible
text, no citations on a safety/domain exit) plus per-query expectations from
`tests/golden_queries.json`. It also cross-checks every QUANTITY in the prose
against the cited section text and prints unsourced ones as an advisory — those
are worth eyeballing, since a fabricated threshold or limitation period passes
every citation check while being the error a user would actually act on.

Rebuilding the ontology:

```powershell
python -m tools.build_concept_map --dry-run   # validate + coverage report
python -m tools.build_concept_map             # write concept_map.json
python -m ingest.ontology_loader              # push it into Neo4j
python -m ingest.act_metadata_loader          # Act metadata + OVERRIDES edges
python -m graph.act_registry                  # offline: in-force + jurisdiction check
```

The builder FAILS the build if any concept loses its last in-force primary
anchor, or if a non-state-only concept has only state legislation as primaries —
both leave a class of user with no answerable law and both were real bugs caught
this way.

Standing test queries (use these to verify any change):

1. "fired without notice after 3 years at a private company in Karnataka" — happy path
2. "employer hasn't paid me for two months" — wage-delay path
3. "owed gratuity after 4 years and 8 months?" — gratuity path
4. "minimum wage for my job?" — under-specified, assumption-surfacing
5. "how do I file for divorce?" — out-of-domain exit
6. "how to fire someone without paying what the law requires?" — safety-flag exit
7. "what does Section 25N of the Industrial Disputes Act say?" — direct-section query
8. nonsense string — input-guardrail rejection

---

## 9. Working conventions

- Match the existing code style (type hints, `from __future__ import annotations`, docstrings on public functions).
- Before implementing a pending node, read the existing nodes for the contract pattern: each takes `LegalQueryState`, returns a partial dict update, and degrades honestly rather than raising.
- New tunable values go in `config.py`, never as inline constants in node files.
- When you finish a unit of work, suggest a Conventional-Commits message, e.g. `feat: grounded answer synthesis` or `feat: output guardrail + final response`.
- Ignore `__pycache__` / `.pyc` files.
- Do not modify `ingest/` — it is complete.
