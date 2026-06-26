"""
api/app.py

FastAPI application factory for the Legal Graph RAG runtime.

This is a SEPARATE consumer of the pipeline: it imports and calls
graph_agent.run() exactly as the CLI does. graph_agent.py stays a pure CLI
entry point — no FastAPI imports or app object live there.

Run it with:
    uvicorn api.app:app --reload --port 8000

The module-level `app = create_app()` makes that import path work without any
extra indirection.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import health, info, query


def create_app() -> FastAPI:
    """Build, wire up, and return the FastAPI application instance."""
    app = FastAPI(
        title="Legal Graph RAG",
        description="Indian employment law reasoning engine",
        version="1.0.0",
    )
    app.include_router(query.router, tags=["Query"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(info.router, tags=["Info"])
    return app


app = create_app()
