# Legal Graph RAG — Architecture Diagrams

Companion to `SYSTEM_EXPLAINED.md`. Three views: the pipeline flow, the full
system stack, and the offline ingestion that builds the graph.

---

## 1. The LangGraph pipeline (control flow)

What happens to one query as it travels the state machine. `[LLM]` = a Gemini
call; `[det]` = deterministic Python/Cypher.

```
                              START
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ entity_extraction        │  [LLM #1]
                   │ raw_query -> extraction   │
                   └─────────────┬───────────┘
                                 ▼
                   ┌─────────────────────────┐
                   │ concept_grounding        │  [det]
                   │ extraction -> grounded    │
                   └─────────────┬───────────┘
                                 ▼
              ┌────────────►┌─────────────────────────┐
              │             │ graph_traversal          │  [det · Neo4j]
              │             │ grounded -> retrieval     │
              │             │  1. direct section lookup │
              │             │     (explicit "Section    │
              │             │     25N" references)      │
              │             │  2. BFS per concept, with  │
              │             │     repealed-Act +         │
              │             │     wrong-state filtering  │
              │             │  3. rank union against     │
              │             │     the query, cap to      │
              │             │     MAX_SECTIONS           │
              │             └─────────────┬───────────┘
              │                           ▼
              │             ┌─────────────────────────┐
              │             │ sufficiency_evaluator    │  [LLM #2]
              │             │ retrieval -> sufficiency  │
              │             └─────────────┬───────────┘
              │                           │
              │              _route_after_sufficiency(state)
              │                           │
              │            ┌──────────────┴───────────────┐
              │   "expand" │                              │ "end"
              │  (not sufficient                          │ (sufficient, OR
              │   AND iters < MAX                         │  iters >= MAX, OR
              │   AND retrieval non-empty)                │  retrieval empty)
              │            ▼                              ▼
              │  ┌─────────────────────┐      ┌─────────────────────────┐
              │  │ graph_expansion      │ [det]│ answer_synthesis         │ [LLM #3]
              │  │ max_hops += 1         │      │ evidence pack ->          │
              │  │ (cap MAX_HOPS_CAP)    │      │ draft_answer +            │
              │  └──────────┬───────────┘      │ cited_section_ids         │
              │             │                  └─────────────┬───────────┘
              └─────────────┘                                ▼
                  (loops back to                ┌─────────────────────────┐
                   graph_traversal)             │ output_guardrail         │ [det · Neo4j]
                                                │ verify citations +        │
                                                │ confidence + disclaimer   │
                                                │ -> verified_section_ids,  │
                                                │    confidence, status     │
                                                └─────────────┬───────────┘
                                                              ▼
                                                ┌─────────────────────────┐
                                                │ final_response           │ [det · never throws]
                                                │ status precedence +       │
                                                │ strip unverified markers  │
                                                │ -> final_answer           │
                                                └─────────────┬───────────┘
                                                              ▼
                                                            END
```

**Key facts**
- The only branch is `_route_after_sufficiency`. The only loop is
  `sufficiency -> expansion -> traversal -> sufficiency`.
- Loop termination is decided by **plain-Python counters** (`MAX_ITERATIONS`),
  never by the LLM.
- `recursion_limit` (25) is a backstop passed at `invoke()` time.
- A `persona` (`citizen` | `lawyer`, set on the initial state at login) rides
  the clipboard through every node but is read only by `answer_synthesis`
  (persona-selected prompt) and `final_response` (persona-aware trailer). It
  changes how the answer reads, never what is retrieved, verified, or cited —
  and adds no LLM call.
- `graph_traversal` is a single node in the graph but does three deterministic
  things internally (see box above): resolve any explicit section reference,
  traverse + filter by temporal/territorial status, then rank and cap. None of
  this adds a node, an edge, or an LLM call — it is all inside
  `agents/nodes/retrieval_node.py`, `graph/traversal.py`, `graph/ranking.py`,
  and `graph/act_registry.py`.
- `final_response` also clears `verified_section_ids` and `confidence_factors`
  back to empty/zero whenever it resolves the status to `out_of_domain` /
  `rejected` / `error` (its own safety-precedence override, not the guardrail's
  `status`) — a refused query must not still be carrying a citation list in
  state for an API consumer to read.

---

## 2. Full system stack (components + data flow)

```
┌──────────────────────────────── YOUR MACHINE ────────────────────────────────┐
│                                                                               │
│   ┌─────────────────────┐      ┌──────────────────────────────────────────┐   │
│   │ uvicorn api.app:app │      │ python graph_agent.py (CLI)              │   │
│   │  POST /query        │      │  interactive persona login + query loop  │   │
│   │  GET  /health       │      └───────────────────┬──────────────────────┘   │
│   │  GET  /graph/stats  │                          │                          │
│   │  GET  /concepts     │                          │ run(query, persona)      │
│   └──────────┬──────────┘                          │                          │
│              │ async run_in_executor                │                          │
│              └─────────────────────┬───────────────┘                          │
│                                    ▼                                          │
│                    GRAPH.invoke({"raw_query":..., "persona":...})              │
│                                                                               │
│  ┌──────────────────── LangGraph workflow (the "agent") ──────────────────┐  │
│  │  STATE = LegalQueryState  (pydantic clipboard, passed node to node)     │  │
│  │                                                                         │  │
│  │   entity_extraction ─► concept_grounding ─► graph_traversal ◄────────┐  │  │
│  │      [LLM #1]            [det]                [det · Neo4j]           │  │  │
│  │                                                    │                 │  │  │
│  │                                            sufficiency_evaluator     │  │  │
│  │                                                [LLM #2]              │  │  │
│  │                                      ┌──────── conditional ──────────┤  │  │
│  │                             "expand" │                          "end"│  │  │
│  │                                      ▼                               │  │  │
│  │                             graph_expansion ─────────────────────────┘  │  │
│  │                                [det]   (loops back, max_hops += 1)       │  │
│  │                                      │ "end"                            │  │
│  │                                      ▼                                  │  │
│  │                             answer_synthesis   [LLM #3]                  │  │
│  │                                      ▼                                  │  │
│  │                             output_guardrail   [det · Neo4j]            │  │
│  │                                      ▼                                  │  │
│  │                             final_response     [det]                    │  │
│  │                                      ▼                                  │  │
│  │                                    END ─► final LegalQueryState          │  │
│  └───────────────────────────┬───────────────────────┬───────────────────┘  │
│                              │ reads (Cypher)         │ builds LLM clients    │
│         ┌────────────────────▼────────────┐    ┌──────▼──────────────────────┐ │
│         │ graph/  (Neo4j access layer)     │    │ agents/llm.py                │ │
│         │   db_connection.py  (driver)     │    │   ChatVertexAI factory       │ │
│         │   queries.py        (all Cypher) │    │   (reads config.py)          │ │
│         │   traversal.py      (BFS engine) │    │   prompts/*.txt  schemas.py  │ │
│         │   ranking.py        (BM25 score) │    │   agents/citations.py        │ │
│         │   act_registry.py   (offline:    │    │   (SHARED marker parse       │ │
│         │     in-force + jurisdiction)     │    │    + strip — one regex)      │ │
│         └────────────────┬────────────────┘    └──────┬──────────────────────┘ │
│                          │ Bolt protocol               │ HTTPS (ADC auth)        │
└──────────────────────────┼──────────────────────────────┼─────────────────────┘
                           ▼                              ▼
                ┌──────────────────────┐        ┌────────────────────┐
                │ Neo4j (Aura)         │        │ Vertex AI          │
                │  5 Act               │        │  Gemini 2.5 Flash  │
                │  255 Section         │        │  (3 call sites)    │
                │  127 CITES           │        └────────────────────┘
                │  45 Concept          │
                │  248 APPLIES_TO      │        ┌────────────────────┐
                │  1 OVERRIDES         │        │ LangSmith (ambient,│
                └──────────────────────┘        │ if env flag on)    │
                                                └────────────────────┘
   config.py (pydantic-settings) feeds ALL layers; .env supplies secrets.
```

**What each new `graph/` module does** (not shown in the box above to keep it
readable):
- `traversal.py` now applies `drop_repealed` / `filter_jurisdiction` **before**
  confidence scoring and BFS expansion — filtering after would let a repealed
  section act as a primary anchor and steer which live sections get retrieved.
- `ranking.py` scores the union of retrieved sections against the query (BM25 +
  relevance tag + hop distance + concept-hit count + act priority). No LLM, no
  network — every signal is derived from data the graph already returned.
- `act_registry.py` reads `data/ontology/act_metadata.json` directly — **not**
  Neo4j — so in-force/jurisdiction lookups and "Section N" reference parsing
  work with zero DB round-trips. `ingest/act_metadata_loader.py` mirrors the
  same facts onto the Act nodes so `graph/queries.py` can join them at read
  time; the JSON file is still the single source of truth.

**Layer summary**

| Layer | Files | Responsibility |
|---|---|---|
| HTTP entry | `api/app.py`, `api/routes/` | FastAPI server — `POST /query`, `GET /health`, `/graph/stats`, `/concepts` |
| CLI entry | `graph_agent.py` | Persona login, interactive query loop, `run(query, persona)` |
| Orchestration | `graph_agent.py` (`build_graph`) | 8 nodes, 1 fork, 1 loop |
| State | `agents/graph_state.py`, `agents/state.py` | The shared clipboard + domain models |
| LLM | `agents/llm.py`, `prompts/`, `schemas.py` | 3 Gemini call sites, structured output; synthesis prompt = `synthesis_base.txt` + persona overlay |
| Citations | `agents/citations.py` | SINGLE marker format — parses (synthesis) AND strips (final response); never duplicate this regex |
| Persona | `agents/persona.py` | Canonical `citizen`/`lawyer` + `normalize_persona`/`match_persona` |
| Graph access | `graph/db_connection.py`, `queries.py`, `traversal.py` | Read-only Neo4j + BFS + temporal/territorial filtering |
| Ranking | `graph/ranking.py` | Deterministic BM25 + graph-signal scoring — no LLM, no network |
| Act metadata | `graph/act_registry.py`, `data/ontology/act_metadata.json` | Offline in-force/jurisdiction facts; section-reference parsing |
| Knowledge | Neo4j DB | The legal graph itself |
| Config | `config.py`, `.env` | Every tunable + secrets |
| Dev tooling | `tools/build_concept_map.py`, `tests/verify.py` | Ontology rebuild + validation; 50-query end-to-end harness (not part of the runtime path) |

---

## 3. Offline ingestion (how the graph gets built)

This runs **before** any query, separately from the agent. Not part of the
runtime path.

```
  data/acts/*.pdf
        │
        ▼
  ingest/pdf_parser.py                  ── extract raw section text
        │
        ▼
  data/processed/sections.jsonl         ── one JSON line per Section
  data/processed/relationships.jsonl    ── one JSON line per CITES edge
        │
        ▼
  ingest/graph_builder.py               ── MERGE nodes + edges into Neo4j
        │   creates: (:Act), (:Section),
        │            (:Act)-[:HAS_SECTION]->(:Section),
        │            (:Section)-[:CITES]->(:Section)
        ▼
  tools/section_concept_map.json        ── curated section -> concept table
  tools/new_concepts.json               ── concepts added beyond the original 25
        │
        ▼
  tools/build_concept_map.py            ── merges + VALIDATES, writes:
        │                                  data/ontology/concept_map.json
        │   fails the build if a concept loses its last in-force primary
        │   anchor, or has only state-legislation primaries (see CLAUDE.md §8)
        ▼
  ingest/ontology_loader.py             ── loads data/ontology/concept_map.json
        │   creates: (:Concept),
        │            (:Section)-[:APPLIES_TO]->(:Concept)  (with relevance)
        ▼
  data/ontology/act_metadata.json       ── per-Act jurisdiction + in-force status
        │
        ▼
  ingest/act_metadata_loader.py         ── annotates existing (:Act) nodes, creates:
        │   SET a.jurisdiction, a.status, a.repealed_by, a.act_priority, ...
        │            (:Act)-[:OVERRIDES {authority}]->(:Act)   (repealer -> repealed)
        ▼
  ┌──────────────────────────────────────────────────┐
  │ Neo4j graph, ready for the agent to read          │
  │   5 Act · 255 Section · 127 CITES                 │
  │   45 Concept · 248 APPLIES_TO · 1 OVERRIDES        │
  └──────────────────────────────────────────────────┘
```

`act_metadata_loader.py` is additive-only: it never creates an Act, only
annotates ones `graph_builder.py` already loaded, and only marks an Act
repealed when the repealing Act is in the corpus and the repeal is backed by a
real `section_id` (`repeal_authority`). Run it any time after
`graph_builder.py`; order relative to `ontology_loader.py` doesn't matter.

---

## 4. The graph schema (what the data looks like)

Node labels and relationship types (from `graph/schema.py`):

```
        ┌──────────┐ HAS_SECTION   ┌──────────┐  APPLIES_TO   ┌──────────┐
        │Act (newer)│──────────────►│ Section  │──────────────►│ Concept  │
        └─────┬────┘               └──────────┘   (relevance:  └──────────┘
              │ OVERRIDES             │     ▲       primary /
              │ {authority: sec_id}   │     │       supporting)
              ▼                       └─────┘
        ┌────────────┐                 CITES
        │Act (repealed)│           (Section -> Section)
        └────────────┘

        ┌──────────┐  INTERPRETS   ┌──────────┐
        │  Ruling  │──────────────►│ Section  │   (v2 — defined, no data yet)
        └──────────┘                └──────────┘
```

| Element | Type | Meaning | Populated by |
|---|---|---|---|
| `Act` | node | A statute (e.g. Industrial Disputes Act 1947). Carries `jurisdiction`, `status` (`in_force`/`repealed`), `in_force_from`, `repealed_by`, `repeal_authority`, `act_priority` | `graph_builder.py` (base fields), `ingest/act_metadata_loader.py` (temporal/territorial fields) |
| `Section` | node | One section within an act (e.g. 25F) | `graph_builder.py` |
| `Concept` | node | Plain-language legal concept (45 total) | `ontology_loader.py` |
| `Ruling` | node | Court ruling (v2, no data) | reserved |
| `HAS_SECTION` | edge | Act -> Section | `graph_builder.py` |
| `CITES` | edge | Section -> Section cross-reference | `graph_builder.py` |
| `APPLIES_TO` | edge | Section -> Concept (carries `relevance`) | `ontology_loader.py` |
| `OVERRIDES` | edge | Act -> Act, newer -> repealed (carries `authority`: the repealing `section_id`). **LIVE, not v2** — 1 edge exists: COW_2019 -> MWA_1948 | `ingest/act_metadata_loader.py` |
| `INTERPRETS` | edge | Ruling -> Section | v2, no data |

**Retrieval uses this shape directly:** grounding lands on a `Concept`,
traversal walks `Concept <-[:APPLIES_TO]- Section` to find anchor sections, then
expands along `Section -[:CITES]-> Section` for connected provisions —
**filtering out any section whose `Act.status = "repealed"` or whose
`Act.jurisdiction` doesn't match the query, before scoring confidence or
expanding.** `OVERRIDES` itself is not walked at query time; it exists so a
human (or a future node) can answer "what replaced this Act?" directly from the
graph. The filtering decision is driven by `Act.status`/`jurisdiction`, which
`act_metadata_loader.py` sets from the same source file that would produce the
`OVERRIDES` edge.
