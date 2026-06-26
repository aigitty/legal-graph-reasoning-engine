"""
api/routes/health.py

GET /health — liveness/readiness probe for the Legal Graph RAG API.

Reports the default persona and pings Neo4j. If the ping fails the route still
returns HTTP 200 with a "degraded" body (honest degradation, per CLAUDE.md §7):
the API process is up even when the graph backend is unreachable.
"""

from __future__ import annotations

from fastapi import APIRouter

from config import cfg
from graph.db_connection import get_driver

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Return pipeline readiness plus a live Neo4j connectivity check."""
    try:
        get_driver().verify_connectivity()
    except Exception as exc:  # noqa: BLE001 — degrade honestly, never raise
        return {"status": "degraded", "detail": str(exc)}

    return {
        "status": "ok",
        "pipeline": "ready",
        "persona_default": cfg.DEFAULT_PERSONA,
    }
