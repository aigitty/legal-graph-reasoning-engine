"""
api/routes/query.py

POST /query — the main entry point. Accepts a QueryRequest, runs the full
LangGraph pipeline via graph_agent.run(), and maps the resulting
LegalQueryState onto a typed QueryResponse.

This route is a thin consumer of run(): it adds ZERO new LLM calls (the three
documented calls all live inside the pipeline) and it never writes to Neo4j.
run() is synchronous (it calls GRAPH.invoke()), so it is dispatched to a thread
pool to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import functools

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agents.persona import normalize_persona
from api.models import ConfidenceFactors, QueryRequest, QueryResponse
from graph_agent import run

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run one legal query end-to-end and return the verified, scored answer."""
    persona = normalize_persona(request.persona)

    try:
        state = await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(run, raw_query=request.query, persona=persona),
        )
    except Exception as exc:  # noqa: BLE001 — never leak to the default handler
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal pipeline error", "error": str(exc)},
        )

    return QueryResponse(
        status=state.status,
        final_answer=state.final_answer,
        confidence=state.confidence,
        confidence_factors=ConfidenceFactors(**state.confidence_factors),
        verified_section_ids=state.verified_section_ids,
        warnings=state.warnings,
        persona=state.persona,
    )
