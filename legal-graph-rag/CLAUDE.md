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
8. **Persona tailors presentation only, never substance.** The selected persona changes the synthesis tone/technicality/structure, how citation markers are *rendered*, and the final-response trailer — it NEVER changes what is retrieved, verified, or allowed to be cited, and it does NOT add an LLM call (rule 1 still holds: exactly three). Removing a *verified* marker from the citizen's visible text is presentation: verification already happened in `output_guardrail_node`, which runs first. Removing an *unverified* marker is rule 5 and is persona-independent.

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
│   └── prompts/            # extraction.txt, sufficiency.txt,
│                           #   synthesis_base.txt (shared integrity rules) +
│                           #   synthesis_citizen.txt / synthesis_lawyer.txt (persona overlays)
│
├── graph/                  # Neo4j access — DO NOT add LLM logic here
│   ├── schema.py           # node/relationship definitions (single source of truth)
│   ├── db_connection.py    # Neo4j driver (reads creds from config)
│   ├── queries.py          # all Cypher (the only place raw Cypher lives)
│   └── traversal.py        # deterministic BFS traversal
│
├── ingest/                 # COMPLETE — DO NOT MODIFY
│   ├── pdf_parser.py
│   ├── graph_builder.py
│   └── ontology_loader.py
│
└── data/
    ├── acts/               # source PDFs
    ├── ontology/concept_map.json   # 25 concepts + aliases — the grounding dictionary
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

- Ingestion layer (`ingest/`) — graph is populated: 5 Acts, 255 Sections, 127 CITES, 25 Concepts, 75 APPLIES_TO.
- Graph layer (`graph/`) — schema, queries, deterministic traversal.
- Full agent pipeline, all three LLM calls + both deterministic guardrail nodes:
  - `synthesis_node` (LLM call 3) — verified working; cites only evidence-pack section_ids, short-circuits the empty-retrieval path with no LLM call. **Persona-aware:** the system prompt is `synthesis_base.txt` + the selected persona overlay (citizen = plain-language/reassuring; lawyer = technical/section-by-section).
  - `output_guardrail_node` — verifies every citation against BOTH the retrieval set (provenance) AND Neo4j (existence), strips failures with a warning, computes the deterministic confidence score (§6), downgrades to `insufficient_evidence` below `MIN_CONFIDENCE`, and injects the disclaimer in code. Degrades honestly if Neo4j is unreachable (provenance-only + warning).
  - `final_response_node` — pure formatting (NO LLM, NO Neo4j), assembles `final_answer` for all five terminal statuses with safety precedence (see §6); strips unverified inline `[SECTION_ID]` markers; wrapped so it never raises. **Persona-aware trailer:** lawyer gets "Verified citations:" (raw ids) + the numeric confidence factor breakdown; citizen gets "The law behind this answer: …" in readable form ("Section 18 of the Code on Wages") and a worded confidence band (high/moderate/low) with no jargon, capped at "moderate" when the evidence is partial.
- `config.py` — pydantic-settings is the single source of all runtime config (model, per-call temps/token budgets, all hard limits, confidence weights, Neo4j creds, GCP project/location). Every former hardcoded constant now reads from here; override any value via an env var of the same name.

**Verified terminal statuses:** `ok`, `insufficient_evidence`, `out_of_domain`, `rejected`, `error` — all exercised (live + offline) and producing correct output.

**Not built yet (the roadmap):**

- `tests/` — unit (confidence formula, citation stripping, grounding, node contracts) + integration (the §8 standing queries against real Neo4j).
- `graph/retrieval.py` — optional refactor of `traversal.py` into parameterized `retrieve()` + `verify_sections()`.
- Clean up `requirements.txt`: it still lists `groq`, `langchain-groq`, and `nemoguardrails`, none of which the runtime uses — the LLM layer is `langchain-google-vertexai` (`ChatVertexAI`, which must be pinned `<4.0.0`; see `agents/llm.py`) and the input/output guardrail logic lives in the pipeline nodes above (NeMo was never wired in). *(The vestigial `guardrails/` folder has already been removed.)*

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
  - `concept_coverage` — grounded ÷ extracted concepts (1.0 if no extraction but something grounded).
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

---

## 7. Validation-failure philosophy

Degrade honestly rather than block. A stripped-citation answer with a warning and lower confidence beats a refusal — EXCEPT for safety flags and out-of-domain queries, which exit before any legal content is generated. Never block on missing user facts (e.g. no tenure/state); proceed and surface assumptions as warnings.

---

## 8. How to run / test

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
