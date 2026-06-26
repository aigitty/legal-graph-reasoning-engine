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

---

## 2. Full system stack (components + data flow)

```
┌──────────────────────────────── YOUR MACHINE ────────────────────────────────┐
│                                                                               │
│   python graph_agent.py  ──►  run(query)  ──►  GRAPH.invoke({"raw_query":...}) │
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
│         └────────────────┬────────────────┘    └──────┬──────────────────────┘ │
│                          │ Bolt protocol               │ HTTPS (ADC auth)        │
└──────────────────────────┼──────────────────────────────┼─────────────────────┘
                           ▼                              ▼
                ┌──────────────────────┐        ┌────────────────────┐
                │ Neo4j (local)        │        │ Vertex AI          │
                │  5 Acts              │        │  Gemini 2.5 Flash  │
                │  255 Section         │        │  (3 call sites)    │
                │  127 CITES           │        └────────────────────┘
                │  25 Concept          │
                │  75 APPLIES_TO       │        ┌────────────────────┐
                └──────────────────────┘        │ LangSmith (ambient,│
                                                │ if env flag on)    │
   config.py (pydantic-settings) feeds ALL      └────────────────────┘
   layers; .env supplies secrets.
```

**Layer summary**

| Layer | Files | Responsibility |
|---|---|---|
| Entry | `graph_agent.py` | Build/compile graph, persona login, `run(query, persona)` |
| Orchestration | `graph_agent.py` (`build_graph`) | 8 nodes, 1 fork, 1 loop |
| State | `agents/graph_state.py`, `agents/state.py` | The shared clipboard + domain models |
| LLM | `agents/llm.py`, `prompts/`, `schemas.py` | 3 Gemini call sites, structured output; synthesis prompt = `synthesis_base.txt` + persona overlay |
| Persona | `agents/persona.py` | Canonical `citizen`/`lawyer` + `normalize_persona`/`match_persona` |
| Graph access | `graph/db_connection.py`, `queries.py`, `traversal.py` | Read-only Neo4j + BFS |
| Knowledge | Neo4j DB | The legal graph itself |
| Config | `config.py`, `.env` | Every tunable + secrets |

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
  ingest/ontology_loader.py             ── loads data/ontology/concept_map.json
        │   creates: (:Concept),
        │            (:Section)-[:APPLIES_TO]->(:Concept)  (with relevance)
        ▼
  ┌──────────────────────────────────────────────────┐
  │ Neo4j graph, ready for the agent to read          │
  │   5 Acts · 255 Section · 127 CITES                │
  │   25 Concept · 75 APPLIES_TO                       │
  └──────────────────────────────────────────────────┘
```

---

## 4. The graph schema (what the data looks like)

Node labels and relationship types (from `graph/schema.py`):

```
        ┌────────┐  HAS_SECTION   ┌──────────┐  APPLIES_TO   ┌──────────┐
        │  Act   │───────────────►│ Section  │──────────────►│ Concept  │
        └────────┘                └──────────┘   (relevance:  └──────────┘
             │                      │     ▲       primary /
             │ OVERRIDES            │     │       supporting)
             │ (Act -> Act, v2)     └─────┘
             ▼                       CITES
        ┌────────┐               (Section -> Section)
        │  Act   │
        └────────┘

        ┌──────────┐  INTERPRETS   ┌──────────┐
        │  Ruling  │──────────────►│ Section  │   (v2 — defined, no data yet)
        └──────────┘                └──────────┘
```

| Element | Type | Meaning | Populated by |
|---|---|---|---|
| `Act` | node | A statute (e.g. Industrial Disputes Act 1947) | `graph_builder.py` |
| `Section` | node | One section within an act (e.g. 25F) | `graph_builder.py` |
| `Concept` | node | Plain-language legal concept (25 total) | `ontology_loader.py` |
| `Ruling` | node | Court ruling (v2, no data) | reserved |
| `HAS_SECTION` | edge | Act -> Section | `graph_builder.py` |
| `CITES` | edge | Section -> Section cross-reference | `graph_builder.py` |
| `APPLIES_TO` | edge | Section -> Concept (carries `relevance`) | `ontology_loader.py` |
| `OVERRIDES` | edge | Act -> Act (newer supersedes older) | manual / v2 |
| `INTERPRETS` | edge | Ruling -> Section | v2 |

**Retrieval uses this shape directly:** grounding lands on a `Concept`,
traversal walks `Concept <-[:APPLIES_TO]- Section` to find anchor sections, then
expands along `Section -[:CITES]-> Section` for connected provisions.
