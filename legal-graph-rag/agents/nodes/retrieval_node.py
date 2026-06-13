"""
agents/nodes/retrieval_node.py

Deterministic graph retrieval node.

Loops over state.grounded_concepts, calls the existing graph.traversal.traverse()
once per concept, and merges the results:
  - sections deduplicated by section_id
  - relevance ("primary"/"supporting") assigned from concept_map.json via
    agents.ontology.relevance_for()
  - CITES edges deduplicated
  - acts_covered as a union

Falls back to raw_query directly if grounding produced no matches, mirroring
main.py's behavior of handing traverse() the raw text and letting it return
an empty result safely if nothing matches.

Still fully DETERMINISTIC: no LLM, no invented sections.
"""

from __future__ import annotations

import logging
from typing import Any

from graph.traversal import traverse
from agents.state import RetrievalResult, SectionContext
from agents.graph_state import LegalQueryState
from agents.ontology import relevance_for

logger = logging.getLogger(__name__)

MAX_HOPS = 2
MAX_SECTIONS = 15


def _normalize_edge(edge: dict[str, Any]) -> dict[str, str]:
    source = edge.get("source_section_id") or edge.get("source_id") or ""
    target = edge.get("target_section_id") or edge.get("target_id") or ""
    return {"source_section_id": str(source), "target_section_id": str(target)}


def _relevance_rank(relevance: str) -> int:
    return 1 if relevance == "primary" else 0


def graph_traversal_node(state: LegalQueryState) -> dict[str, Any]:
    concepts = state.grounded_concepts or (
        [state.raw_query.strip().lower()] if state.raw_query.strip() else []
    )

    if not concepts:
        return {"error": "No grounded concepts and no raw_query to fall back to."}

    relevance_map = relevance_for(concepts)

    sections_by_id: dict[str, SectionContext] = {}
    edges_seen: dict[tuple[str, str], dict[str, str]] = {}
    acts_covered: list[str] = []
    acts_seen: set[str] = set()
    confidences: list[float] = []
    any_non_empty = False

    for concept in concepts:
        try:
            tr = traverse(concept_name=concept, max_hops=MAX_HOPS, max_sections=MAX_SECTIONS,exact=True)
        except Exception as exc:
            logger.exception("Graph traversal failed for concept %r", concept)
            return {"error": f"Graph traversal failed for {concept!r}: {exc}"}

        confidences.append(float(getattr(tr, "confidence", 0.0) or 0.0))
        if not getattr(tr, "is_empty", True):
            any_non_empty = True

        for raw_section in (getattr(tr, "all_sections", []) or []):
            section_id = str(raw_section.get("section_id") or raw_section.get("id") or "")
            sec = SectionContext.from_dict(
                data=raw_section,
                relevance=relevance_map.get(section_id, "supporting"),
                source_concept=concept,
            )

            existing = sections_by_id.get(sec.section_id)
            if existing is None:
                sections_by_id[sec.section_id] = sec
                if sec.act_id and sec.act_id not in acts_seen:
                    acts_seen.add(sec.act_id)
                    acts_covered.append(sec.act_id)
            else:
                if _relevance_rank(sec.relevance) > _relevance_rank(existing.relevance):
                    existing.relevance = sec.relevance
                if concept not in existing.source_concept.split(", "):
                    existing.source_concept = f"{existing.source_concept}, {concept}"

        for raw_edge in (getattr(tr, "cites_edges", None) or []):
            edge = _normalize_edge(raw_edge)
            edges_seen[(edge["source_section_id"], edge["target_section_id"])] = edge

    result = RetrievalResult(
        concepts_queried=concepts,
        sections=list(sections_by_id.values()),
        cites_edges=list(edges_seen.values()),
        total_found=len(sections_by_id),
        acts_covered=acts_covered,
        confidence=max(confidences) if confidences else 0.0,
        is_empty=not any_non_empty,
        jurisdiction_applied="Central",
    )
    return {"retrieval": result}