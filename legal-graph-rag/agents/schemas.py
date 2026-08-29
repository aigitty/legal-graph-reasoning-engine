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
    monthly_salary: Optional[float] = Field(
        default=None,
        description="The employee's MONTHLY salary/wage in INR, as a plain "
        "number with no currency symbol or commas, if stated or clearly "
        "computable (e.g. an annual figure divided by 12). E.g. '25000 a "
        "month' -> 25000.0; '3 lakhs a year' -> 25000.0. Null if no figure is "
        "given — never guess a salary the query does not support.",
    )
    triggering_event: Optional[str] = Field(
        default=None,
        description="Short description of what happened, e.g. 'terminated without notice'.",
    )
    event_date: Optional[str] = Field(
        default=None,
        description="WHEN the triggering event happened, if the query states or "
        "clearly implies a date, month, or year — normalized to 'YYYY-MM-DD', "
        "'YYYY-MM', or 'YYYY' (whichever precision the query supports). E.g. "
        "'fired on 3 August 2025' -> '2025-08-03'; 'let go last August' -> "
        "'2025-08' (infer the year from context/today's date if unstated but "
        "implied by a relative phrase). Null if no date is stated or implied — "
        "do NOT default to today's date.",
    )
    # NOTE: these descriptions are part of the JSON schema sent to Gemini, so
    # they are prompt text, not comments. This one used to enumerate the corpus
    # by name — and named three Acts that are now repealed while omitting the
    # two Codes that replaced them. Judging the SUBJECT rather than the statute
    # list is both more accurate and immune to that drift; the authoritative
    # corpus is injected into extraction.txt from act_metadata.json.
    in_domain: bool = Field(
        description="True if the query is about Indian employment or labour law as "
        "a SUBJECT — wages, hours, leave, termination, retrenchment, gratuity, "
        "provident fund, ESI, maternity benefit, bonus, deductions, workplace "
        "disputes, and the remedies for them. False only for a genuinely different "
        "subject (criminal, family, tax, property, immigration, nonsense). Do not "
        "set false merely because this system may not hold the exact statute."
    )
    safety_flag: Optional[str] = Field(
        default=None,
        description="Set to a SHORT REASON only if the request seeks to help an "
        "employer evade legal obligations or harm an employee. Otherwise omit it "
        "entirely or return null — never the strings 'none', 'null' or 'false'.",
    )

    @property
    def is_unsafe(self) -> bool:
        """
        Whether the safety flag genuinely fired.

        safety_flag is a free-text REASON, so plain truthiness treats any string
        as a refusal — including the sentinel words a model reaches for when it
        means "nothing to report". A query wrongly classified here is refused
        outright with no legal content, which is the most expensive possible
        failure for a user, so the sentinels are filtered explicitly rather than
        trusted not to appear.
        """
        flag = (self.safety_flag or "").strip().strip(".").lower()
        return bool(flag) and flag not in {
            "none", "null", "nil", "false", "no", "n/a", "na", "-", "not applicable",
        }


class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved sections are enough to answer the query."
    )
    missing: str = Field(
        default="",
        description="If insufficient, a brief description of what legal coverage "
        "is missing. Empty string if sufficient.",
    )