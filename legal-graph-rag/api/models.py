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
    """
    The four weighted components behind the deterministic confidence score.

    All default to 0.0 because final_response_node CLEARS the factor breakdown
    on the no-content terminal statuses (out_of_domain / rejected / error) — it
    scores an answer that is not being given. Without defaults, the route's
    `ConfidenceFactors(**state.confidence_factors)` raises a ValidationError on
    the empty dict and turns an honest refusal into a 500.
    """

    concept_coverage: float = 0.0
    seed_strength: float = 0.0
    sufficiency_score: float = 0.0
    citation_validity: float = 0.0


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
