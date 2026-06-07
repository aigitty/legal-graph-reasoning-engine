"""
agents/state.py

Single source of truth for all typed data contracts used by the Legal Graph RAG
agent pipeline.

This module defines the dataclasses that flow between agents:

User Query
    -> ExtractedEntities
    -> RetrievalResult
    -> LegalAnswer
    -> ValidatedAnswer

Rules:
- No raw dictionaries should be passed between agents.
- Agent files, helper files, API routes, and UI layers should import contracts
  from this module.
- This file has zero external dependencies: no ADK, no Gemini, no Neo4j,
  no LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SectionContext:
    """
    Represents one legal section as it flows through the agent pipeline.

    Produced by:
        Agent 2 — graph_retriever

    This is the unit of legal knowledge passed from the graph layer into the
    reasoning layer. It wraps raw Neo4j section data with retrieval metadata
    such as relevance, source concept, and ranking priority.
    """

    section_id: str = ""
    section_number: str = ""
    section_title: str = ""
    section_text: str = ""
    act_id: str = ""
    act_name: str = ""
    relevance: str = "supporting"
    source_concept: str = ""
    act_priority: int = 0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        relevance: str,
        source_concept: str,
    ) -> "SectionContext":
        """
        Build SectionContext from a raw graph dictionary.

        This method is the bridge between graph/queries.py and the agent layer.
        It handles missing keys gracefully so graph changes do not immediately
        crash the agent pipeline.
        """

        return cls(
            section_id=str(data.get("section_id", "")),
            section_number=str(data.get("section_number", "")),
            section_title=str(data.get("section_title", "")),
            section_text=str(data.get("section_text", "")),
            act_id=str(data.get("act_id", "")),
            act_name=str(data.get("act_name", "")),
            relevance=relevance or "supporting",
            source_concept=source_concept or "",
            act_priority=int(data.get("act_priority", 0) or 0),
        )


@dataclass
class ExtractedEntities:
    """
    Output of Agent 1 — entity_extractor.

    Represents the structured legal situation extracted from the user's raw
    natural language query.
    """

    raw_query: str = ""
    concepts: list[str] = field(default_factory=list)
    jurisdiction: str = "Central"
    entity_type: str = "employee"
    years_of_service: float | None = None
    employment_type: str | None = None
    confidence: float = 0.0
    is_valid: bool = True
    validation_note: str = ""


@dataclass
class RetrievalResult:
    """
    Output of Agent 2 — graph_retriever.

    Contains all legal sections, graph edges, jurisdiction handling, and
    retrieval confidence produced by Neo4j graph traversal.
    """

    concepts_queried: list[str] = field(default_factory=list)
    sections: list[SectionContext] = field(default_factory=list)
    cites_edges: list[dict[str, str]] = field(default_factory=list)
    total_found: int = 0
    acts_covered: list[str] = field(default_factory=list)
    confidence: float = 0.0
    is_empty: bool = True
    jurisdiction_applied: str = "Central"

    @property
    def section_ids(self) -> set[str]:
        """Return all retrieved section IDs for fast citation validation."""
        return {section.section_id for section in self.sections if section.section_id}

    @property
    def primary_sections(self) -> list[SectionContext]:
        """Return only sections marked as primary."""
        return [
            section
            for section in self.sections
            if section.relevance.lower() == "primary"
        ]

    @property
    def formatted_acts(self) -> str:
        """Return a human-readable list of unique act names covered."""
        act_names: list[str] = []
        seen: set[str] = set()

        for section in self.sections:
            if section.act_name and section.act_name not in seen:
                seen.add(section.act_name)
                act_names.append(section.act_name)

        return " · ".join(act_names)


@dataclass
class LegalAnswer:
    """
    Output of Agent 3 — legal_reasoner.

    Represents Gemini's structured legal reasoning over retrieved graph context.
    Every cited section must come from RetrievalResult.section_ids.
    """

    raw_query: str = ""
    applicable_sections: list[str] = field(default_factory=list)
    summary: str = ""
    user_rights: list[str] = field(default_factory=list)
    recommended_action: str = ""
    relevant_acts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    confidence_reason: str = ""
    disclaimer: str = ""
    reasoning_path: list[str] = field(default_factory=list)


@dataclass
class CitationCheck:
    """
    Internal guardrails citation verification result.

    Produced by:
        Agent 4 — guardrails_validator helper logic

    This is not passed between agents directly, but is stored inside the final
    ValidatedAnswer for auditability.
    """

    passed: bool = False
    valid_ids: set[str] = field(default_factory=set)
    invalid_ids: list[str] = field(default_factory=list)
    check_timestamp: str = ""


@dataclass
class ValidatedAnswer:
    """
    Final output of the entire Legal Graph RAG pipeline.

    Produced by:
        Agent 4 — guardrails_validator

    This is what graph_agent.run(), FastAPI routes, and UI layers should return.
    The UI should render only final_response.
    """

    answer: LegalAnswer = field(default_factory=LegalAnswer)
    citation_check: CitationCheck = field(default_factory=CitationCheck)
    safety_passed: bool = False
    safety_notes: list[str] = field(default_factory=list)
    final_response: str = ""
    was_modified: bool = False
    modification_notes: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    agent_versions: dict[str, str] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        """Return True only when citation, safety, and confidence checks pass."""
        return (
            self.citation_check.passed
            and self.safety_passed
            and self.answer.confidence >= 0.7
        )

    @property
    def cited_section_count(self) -> int:
        """Return the number of sections cited by the legal answer."""
        return len(self.answer.applicable_sections)


# ------------------------------------------------------------------
# Smoke Test / Living Example
# ------------------------------------------------------------------
# This block is NOT used by the runtime agent pipeline.
# It exists only to validate contracts and demonstrate
# real-world data flow through the system.
# ------------------------------------------------------------------


if __name__ == "__main__":
    raw_query = "I was fired without notice after 3 years in Karnataka"

    entities = ExtractedEntities(
        raw_query=raw_query,
        concepts=["wrongful termination", "notice period"],
        jurisdiction="Karnataka",
        entity_type="employee",
        years_of_service=3.0,
        employment_type="private",
        confidence=0.95,
        is_valid=True,
        validation_note="",
    )

    section_25f = SectionContext(
        section_id="IDA_1947_S25F",
        section_number="25F",
        section_title="Conditions precedent to retrenchment of workmen",
        section_text=(
            "No workman employed in any industry who has been in continuous "
            "service for not less than one year under an employer shall be "
            "retrenched until the workman has been given one month's notice "
            "or wages in lieu of such notice and retrenchment compensation."
        ),
        act_id="IDA_1947",
        act_name="Industrial Disputes Act, 1947",
        relevance="primary",
        source_concept="wrongful termination",
        act_priority=2,
    )

    section_25g = SectionContext(
        section_id="IDA_1947_S25G",
        section_number="25G",
        section_title="Procedure for retrenchment",
        section_text=(
            "Where any workman in an industrial establishment is to be "
            "retrenched and he belongs to a particular category of workmen, "
            "the employer shall ordinarily retrench the workman who was the "
            "last person to be employed in that category."
        ),
        act_id="IDA_1947",
        act_name="Industrial Disputes Act, 1947",
        relevance="supporting",
        source_concept="wrongful termination",
        act_priority=2,
    )

    section_2a = SectionContext(
        section_id="IDA_1947_S2A",
        section_number="2A",
        section_title=(
            "Dismissal, etc., of an individual workman to be deemed to be "
            "an industrial dispute"
        ),
        section_text=(
            "Where any employer discharges, dismisses, retrenches, or "
            "otherwise terminates the services of an individual workman, "
            "any dispute connected with such termination shall be deemed "
            "to be an industrial dispute."
        ),
        act_id="IDA_1947",
        act_name="Industrial Disputes Act, 1947",
        relevance="supporting",
        source_concept="wrongful termination",
        act_priority=2,
    )

    retrieval = RetrievalResult(
        concepts_queried=entities.concepts,
        sections=[section_25f, section_25g, section_2a],
        cites_edges=[
            {
                "source_section_id": "IDA_1947_S25F",
                "target_section_id": "IDA_1947_S25G",
            },
            {
                "source_section_id": "IDA_1947_S2A",
                "target_section_id": "IDA_1947_S25F",
            },
        ],
        total_found=3,
        acts_covered=["IDA_1947"],
        confidence=0.92,
        is_empty=False,
        jurisdiction_applied="Karnataka",
    )

    answer = LegalAnswer(
        raw_query=raw_query,
        applicable_sections=["IDA_1947_S25F", "IDA_1947_S25G", "IDA_1947_S2A"],
        summary=(
            "Based on the retrieved Industrial Disputes Act sections, termination "
            "without notice after three years of service may raise retrenchment "
            "and industrial dispute issues if you qualify as a workman."
        ),
        user_rights=[
            "You may be entitled to one month's notice or wages in lieu of notice [Section 25F, IDA 1947].",
            "You may be entitled to retrenchment compensation if the statutory conditions apply [Section 25F, IDA 1947].",
            "You may be able to raise an individual industrial dispute regarding termination [Section 2A, IDA 1947].",
        ],
        recommended_action=(
            "Collect your appointment letter, termination communication, salary "
            "records, and service proof, then consult a labour lawyer or approach "
            "the appropriate labour authority."
        ),
        relevant_acts=["Industrial Disputes Act, 1947"],
        confidence=0.82,
        confidence_reason=(
            "Primary and supporting sections were retrieved for wrongful termination, "
            "but final applicability depends on whether the user legally qualifies as a workman."
        ),
        disclaimer=(
            "This is general legal information based on retrieved statutory text, "
            "not legal advice. Please consult a qualified lawyer for advice on your case."
        ),
        reasoning_path=[
            "Identified the query as an employment termination issue.",
            "Mapped the situation to wrongful termination and notice period concepts.",
            "Retrieved Section 25F as the primary retrenchment provision.",
            "Retrieved Section 25G as supporting procedure for retrenchment.",
            "Retrieved Section 2A because individual termination disputes may be treated as industrial disputes.",
        ],
    )

    citation_check = CitationCheck(
        passed=True,
        valid_ids={"IDA_1947_S25F", "IDA_1947_S25G", "IDA_1947_S2A"},
        invalid_ids=[],
        check_timestamp=datetime.now().isoformat(timespec="seconds"),
    )

    validated = ValidatedAnswer(
        answer=answer,
        citation_check=citation_check,
        safety_passed=True,
        safety_notes=[],
        final_response=(
            "Based on the retrieved graph context, your termination may involve "
            "rights under the Industrial Disputes Act, 1947, especially Sections "
            "25F, 25G, and 2A. This depends on whether you qualify as a workman. "
            "Please consult a qualified labour lawyer before taking action."
        ),
        was_modified=False,
        modification_notes=[],
        processing_time_ms=184.7,
        agent_versions={
            "entity_extractor": "gemini-2.0-flash",
            "graph_retriever": "no-llm",
            "legal_reasoner": "gemini-2.0-pro",
            "guardrails_validator": "rule-based",
        },
    )

    print("\n=== ExtractedEntities ===")
    print(f"Query: {entities.raw_query}")
    print(f"Concepts: {entities.concepts}")
    print(f"Jurisdiction: {entities.jurisdiction}")
    print(f"Valid: {entities.is_valid}, Confidence: {entities.confidence}")

    print("\n=== SectionContext ===")
    print(f"Primary Section: {section_25f.section_id}")
    print(f"Title: {section_25f.section_title}")
    print(f"Act: {section_25f.act_name}")
    print(f"Relevance: {section_25f.relevance}")

    print("\n=== RetrievalResult ===")
    print(f"Sections found: {retrieval.total_found}")
    print(f"Section IDs: {sorted(retrieval.section_ids)}")
    print(f"Primary sections: {[s.section_id for s in retrieval.primary_sections]}")
    print(f"Acts covered: {retrieval.formatted_acts}")

    print("\n=== LegalAnswer ===")
    print(f"Summary: {answer.summary}")
    print(f"Applicable sections: {answer.applicable_sections}")
    print(f"Confidence: {answer.confidence}")
    print(f"Confidence reason: {answer.confidence_reason}")

    print("\n=== CitationCheck ===")
    print(f"Passed: {citation_check.passed}")
    print(f"Valid IDs: {sorted(citation_check.valid_ids)}")
    print(f"Invalid IDs: {citation_check.invalid_ids}")

    print("\n=== ValidatedAnswer ===")
    print(f"Reliable: {validated.is_reliable}")
    print(f"Cited section count: {validated.cited_section_count}")
    print(f"Processing time: {validated.processing_time_ms} ms")
    print(f"Final response: {validated.final_response}")