# Legal Graph RAG — Indian Employment Law Reasoning Engine

A **Graph RAG** system that answers plain-language Indian employment law questions
by deterministically traversing a Neo4j knowledge graph and using an LLM only to
synthesise a grounded answer from retrieved statutory text.

Every section in the answer was physically retrieved from the graph and re-verified
against Neo4j before display. The LLM never invents a section.

---

## What it covers

Five Indian labour statutes ingested; **four currently in force**:

| Act | Short ID | Status |
|---|---|---|
| Industrial Disputes Act, 1947 | `IDA_1947` | In force |
| Payment of Gratuity Act, 1972 | `PGA_1972` | In force |
| Code on Wages, 2019 | `COW_2019` | In force |
| Karnataka Shops and Commercial Establishments Act, 1961 | `KSEA_1961` | In force (Karnataka only) |
| Minimum Wages Act, 1948 | `MWA_1948` | **Repealed** by Code on Wages, 2019 §69 |

The engine tracks in-force status and territorial reach for every Act
(`data/ontology/act_metadata.json`) and enforces both deterministically at
retrieval time: `MWA_1948` is never cited (superseded by `COW_2019`), and
`KSEA_1961` is only cited to users in Karnataka.

**255 Sections · 127 CITES edges · 45 Concepts · 248 APPLIES_TO edges · 1 OVERRIDES edge**

---

## Stack

- **Python 3.10**
- **LangGraph** — single state-machine workflow (8 nodes, 1 conditional fork, 1 loop)
- **LangChain + Gemini 2.5 Flash via Vertex AI** — exactly 3 LLM call sites
- **Neo4j** — knowledge graph (read-only at runtime)
- **FastAPI + uvicorn** — HTTP API
- **LangSmith** — ambient tracing (set env vars, traces flow automatically)

---

## Quickstart

```powershell
# activate the virtualenv
.\venv\Scripts\activate

# interactive CLI (prompts for persona, then query loop)
python graph_agent.py

# HTTP API server — docs at http://localhost:8000/docs
uvicorn api.app:app --reload --port 8000

# raw graph debugger — no LLM, no agents
python main.py
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/query` | Run a legal query end-to-end |
| `GET` | `/health` | Neo4j ping + readiness |
| `GET` | `/graph/stats` | Live node/edge counts |
| `GET` | `/concepts` | All 25 grounded concepts |
| `GET` | `/docs` | Interactive Swagger UI |

**Example request:**
```json
POST /query
{
  "query": "I was fired without notice after 3 years at a private company in Karnataka.",
  "persona": "citizen"
}
```

`persona` accepts `"citizen"` (plain-language, reassuring) or `"lawyer"` / `"judge"` /
`"advocate"` (technical, section-by-section). Defaults to `"citizen"`.

**Example response shape:**
```json
{
  "status": "ok",
  "final_answer": "...",
  "confidence": 0.95,
  "confidence_factors": {
    "concept_coverage": 1.0,
    "seed_strength": 1.0,
    "sufficiency_score": 1.0,
    "citation_validity": 0.82
  },
  "verified_section_ids": ["IDA_1947_S25F", "IDA_1947_S25N"],
  "warnings": [],
  "persona": "citizen"
}
```

Terminal statuses: `ok` · `insufficient_evidence` · `out_of_domain` · `rejected` · `error`

---

## Environment setup

Create a `.env` file at the project root:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Google Cloud / Vertex AI (auth via ADC — run: gcloud auth application-default login)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# LangSmith (optional — remove to disable tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=legal-graph-rag
```

Do **not** set `GOOGLE_APPLICATION_CREDENTIALS` — auth is via ADC.

---

## Pipeline

```
extraction [LLM #1] → grounding [det] → traversal [det · Neo4j]
    → sufficiency [LLM #2] → [conditional]
         "expand" → expansion [det] → traversal        (loop, ≤ 2 iterations)
         "end"    → synthesis [LLM #3] → output_guardrail [det · Neo4j]
                       → final_response [det] → END
```

**Three LLM calls. Everything else is deterministic Python or Cypher.**

The confidence score is a deterministic weighted sum of four factors:
`0.35·concept_coverage + 0.25·seed_strength + 0.20·sufficiency_score + 0.20·citation_validity`

---

## Known limitations

- Maternity Benefit Act and other statutes not ingested → queries about maternity
  leave return `insufficient_evidence` honestly.
- No early-exit for safety-flag queries that ground to real concepts (they spend a
  synthesis LLM call whose output is discarded by `final_response`).
- Tests not yet written.

---

## Docs

- `CLAUDE.md` — project rules and working conventions for Claude Code
- `HOW_IT_WORKS_SIMPLE.md` — plain-English walkthrough of one query through the pipeline
- `SYSTEM_EXPLAINED.md` — deep technical walkthrough (LangGraph, Cypher, config)
- `ARCHITECTURE_DIAGRAM.md` — pipeline, stack, and ingestion diagrams
