"""
agents/graph_state.py

LangGraph channel state for the Legal Graph RAG agent workflow.

This is deliberately separate from agents/state.py:
- agents/state.py      -> dependency-free DOMAIN dataclasses (RetrievalResult, etc.)
- agents/graph_state.py -> the Pydantic ORCHESTRATION state that LangGraph
  passes from node to node.

Step 5 added:
  - max_hops: current traversal depth, mutated by graph_expansion_node

Step 6 adds:
  - draft_answer: free-text answer produced by answer_synthesis_node
  - cited_section_ids: [SECTION_ID] markers parsed out of draft_answer
    (UNVALIDATED here — existence/provenance checks belong to the output
    guardrail step, not synthesis)

Step 7 (output guardrail + final response) adds:
  - verified_section_ids: citations that passed BOTH the retrieval-set and
    Neo4j-existence checks in output_guardrail_node (the only ids allowed to
    reach the user, per CLAUDE.md rules 3 & 5)
  - confidence: deterministic score in [0, 1] (CLAUDE.md section 6 formula)
  - confidence_factors: the four weighted components behind `confidence`
  - status: terminal classification — "ok" | "insufficient_evidence" |
    "out_of_domain" | "rejected" | "error"
  - disclaimer: legal disclaimer injected deterministically in code (NEVER
    by the LLM)
  - final_answer: the fully assembled, user-facing response (final_response_node)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.state import RetrievalResult
from agents.schemas import ExtractionResult, SufficiencyVerdict


class LegalQueryState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_query: str = ""
    # Persona selected at login — tailors how the answer is written, never what
    # is retrieved or cited. Canonical values in agents/persona.py; empty/unknown
    # normalizes to "citizen" at the point of use (synthesis & final response).
    persona: str = ""
    concept: str = ""  # legacy direct-concept override; superseded by grounding
    extraction: Optional[ExtractionResult] = None
    grounded_concepts: list[str] = Field(default_factory=list)
    # Extracted phrases that matched NO concept. Needed because
    # grounded_concepts is a union: it cannot tell you whether every extracted
    # phrase was covered or whether one phrase produced several concepts while
    # another produced none. output_guardrail_node scores concept_coverage from
    # this (CLAUDE.md section 6).
    ungrounded_phrases: list[str] = Field(default_factory=list)
    max_hops: int = 2
    retrieval: Optional[RetrievalResult] = None
    sufficiency: Optional[SufficiencyVerdict] = None
    retrieval_iterations: int = 0
    draft_answer: str = ""
    cited_section_ids: list[str] = Field(default_factory=list)

    # Step 7 — output guardrail + final response
    verified_section_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    status: str = ""
    disclaimer: str = ""
    final_answer: str = ""

    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None