"""
api/models.py

Pydantic request/response contracts for the FastAPI layer.

These models are the API's typed surface. They keep the boundary fully typed
end-to-end (no raw dicts cross it) and decouple HTTP clients from the internal
LegalQueryState shape. Map LegalQueryState -> QueryResponse inside the route
handler; never expose the orchestration state directly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import cfg


class QueryRequest(BaseModel):
    """A single legal question plus the persona the answer is tailored for."""

    query: str = Field(..., description="The raw, plain-language legal question.")
    persona: str = Field(
        default=cfg.DEFAULT_PERSONA,
        description='Audience persona: "citizen" | "lawyer" (anything else '
        'normalizes to "citizen").',
    )


class ConfidenceFactors(BaseModel):
    """The four weighted components behind the deterministic confidence score."""

    concept_coverage: float
    seed_strength: float
    sufficiency_score: float
    citation_validity: float


class QueryResponse(BaseModel):
    """The fully assembled, user-facing result of one pipeline run."""

    status: str = Field(
        ...,
        description="Terminal status: ok | insufficient_evidence | "
        "out_of_domain | rejected | error.",
    )
    final_answer: str
    confidence: float
    confidence_factors: ConfidenceFactors
    verified_section_ids: list[str]
    warnings: list[str]
    persona: str
