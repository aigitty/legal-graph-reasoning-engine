"""
agents/schemas.py

Pydantic schemas for structured Gemini outputs.

ExtractionResult is the output of entity_extraction_node — the first
Gemini call in the pipeline. in_domain / safety_flag are captured here
(per architecture: bundled into the extraction call) but are NOT used for
conditional routing yet — that arrives with the guardrails step.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    legal_concepts: list[str] = Field(
        default_factory=list,
        description="Short plain-language legal concept phrases implied by the query "
        "(e.g. 'wrongful termination', 'notice period', 'salary not paid').",
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Indian state explicitly mentioned or clearly implied by context, else null.",
    )
    employment_type: Optional[str] = Field(
        default=None,
        description="Type of employer if stated, e.g. 'private company', 'factory', 'shop'.",
    )
    years_of_service: Optional[float] = Field(
        default=None,
        description="Years of continuous service if stated or clearly implied.",
    )
    triggering_event: Optional[str] = Field(
        default=None,
        description="Short description of what happened, e.g. 'terminated without notice'.",
    )
    in_domain: bool = Field(
        description="True only if the query concerns Indian employment/labour law "
        "covered by the Industrial Disputes Act 1947, Payment of Gratuity Act 1972, "
        "Minimum Wages Act 1948, Karnataka Shops and Establishments Act 1961, or "
        "Code on Wages 2019."
    )
    safety_flag: Optional[str] = Field(
        default=None,
        description="Set only if the request seeks to help an employer evade legal "
        "obligations or harm an employee; otherwise null.",
    )
    
class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved sections are enough to answer the query."
    )
    missing: str = Field(
        default="",
        description="If insufficient, a brief description of what legal coverage "
        "is missing. Empty string if sufficient.",
    )