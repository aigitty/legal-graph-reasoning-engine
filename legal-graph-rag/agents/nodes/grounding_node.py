"""
agents/nodes/grounding_node.py

Deterministic concept-grounding node.

Grounds each phrase in state.extraction.legal_concepts (if present) against
data/ontology/concept_map.json via agents.ontology.ground_query(), unioning
matches. Falls back to grounding state.raw_query directly if extraction
produced no usable concepts. No LLM here.
"""

from __future__ import annotations

from agents.graph_state import LegalQueryState
from agents.ontology import ground_query


def concept_grounding_node(state: LegalQueryState) -> dict:
    candidates: list[str] = []
    if state.extraction and state.extraction.legal_concepts:
        candidates = state.extraction.legal_concepts

    if not candidates:
        candidates = [state.raw_query]

    grounded: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        for concept_name in ground_query(text):
            if concept_name not in seen:
                seen.add(concept_name)
                grounded.append(concept_name)

    if not grounded:
        return {
            "grounded_concepts": [],
            "warnings": state.warnings + [
                f"No concept matched for candidates: {candidates!r}"
            ],
        }

    return {"grounded_concepts": grounded}