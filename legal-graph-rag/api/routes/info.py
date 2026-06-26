"""
api/routes/info.py

Read-only introspection routes:
  GET /graph/stats — live node/edge counts from Neo4j.
  GET /concepts    — every Concept node as {id, name}.

Both delegate to helper functions in graph/queries.py — no raw Cypher lives
here (CLAUDE.md: queries.py is the only place raw Cypher exists).
"""

from __future__ import annotations

from fastapi import APIRouter

from graph.queries import get_all_concepts, get_graph_stats

router = APIRouter()


@router.get("/graph/stats")
def graph_stats() -> dict:
    """Return live node counts (by label) and edge counts (by type)."""
    return get_graph_stats()


@router.get("/concepts")
def concepts() -> dict:
    """Return every grounded Concept node as an {id, name} list."""
    return {"concepts": get_all_concepts()}
