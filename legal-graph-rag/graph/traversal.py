from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from graph.queries import (
    get_sections_for_concept,
    get_sections_for_exact_concept,
    get_neighbors,
    get_subgraph,
)

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    concept_queried: str
    anchor_sections: list[dict]
    expanded_sections: list[dict]
    all_sections: list[dict]
    cites_edges: list[dict]
    concepts_matched: list[str]
    hops_taken: int
    confidence: float
    is_empty: bool


def _section_id(section: dict) -> str | None:
    return section.get("section_id") or section.get("id")


def _sort_section_key(section: dict) -> tuple[str, str]:
    return (
        str(section.get("act_id") or ""),
        str(section.get("section_number") or ""),
    )


def _find_anchors(
    concept_name: str, exact: bool = False
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Find primary and supporting section anchors for a concept.

    Parameters:
        concept_name: Plain-language legal concept extracted by the agent layer.
        exact: When True, match the concept by exact canonical name
               (agent layer already grounded it). When False, use the
               looser CONTAINS matching (main.py / raw-text callers).

    Returns:
        A tuple containing:
        - primary_sections: Section nodes marked as primary anchors.
        - supporting_sections: Section nodes marked as supporting anchors.
        - concepts_matched: Concept names matched in the graph.
    """
    rows = (
        get_sections_for_exact_concept(concept_name)
        if exact
        else get_sections_for_concept(concept_name)
    ) or []

    primary_sections: list[dict] = []
    supporting_sections: list[dict] = []
    concepts_matched: list[str] = []

    for row in rows:
        section = row.get("section", row)
        relation_type = (
            row.get("match_type")
            or row.get("relationship_type")
            or row.get("anchor_type")
            or row.get("type")
            or ""
        )
        matched_concept = row.get("concept_name") or row.get("concept")

        if matched_concept and matched_concept not in concepts_matched:
            concepts_matched.append(matched_concept)

        if str(relation_type).lower() == "supporting":
            supporting_sections.append(section)
        else:
            primary_sections.append(section)

    return primary_sections, supporting_sections, concepts_matched


def _expand_from_anchors(
    anchor_section_ids: list[str],
    max_hops: int,
) -> tuple[list[dict], int]:
    """
    Expand from primary anchor sections using BFS over CITES relationships.

    Parameters:
        anchor_section_ids: Section IDs used as traversal starting points.
        max_hops: Maximum number of expansion rounds.

    Returns:
        A tuple containing:
        - expanded_sections: Newly discovered Section nodes, excluding anchors.
        - hops_taken: Number of hop rounds actually performed.
    """
    if max_hops <= 0 or not anchor_section_ids:
        return [], 0

    visited = set(anchor_section_ids)
    frontier = set(anchor_section_ids)
    expanded_sections: list[dict] = []
    hops_taken = 0

    for hop in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        new_sections_this_hop: list[dict] = []

        for section_id in frontier:
            neighbors = get_neighbors(section_id) or []

            for neighbor in neighbors:
                neighbor_section = neighbor.get("node", neighbor)
                labels = neighbor_section.get("labels", [])

                if labels and "Section" not in labels:
                    continue

                neighbor_id = _section_id(neighbor_section)
                if not neighbor_id or neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                next_frontier.add(neighbor_id)
                new_sections_this_hop.append(neighbor_section)

        logger.debug(
            "Traversal hop %s completed. New sections found: %s",
            hop,
            len(new_sections_this_hop),
        )

        if not new_sections_this_hop:
            break

        expanded_sections.extend(new_sections_this_hop)
        frontier = next_frontier
        hops_taken = hop

    return expanded_sections, hops_taken


def _score_confidence(primary_count: int, supporting_count: int) -> float:
    """
    Compute traversal confidence from anchor availability.

    Parameters:
        primary_count: Number of primary anchor sections.
        supporting_count: Number of supporting anchor sections.

    Returns:
        1.0 when primary anchors exist, 0.6 when only supporting anchors exist,
        otherwise 0.0.
    """
    if primary_count > 0:
        return 1.0
    if supporting_count > 0:
        return 0.6
    return 0.0


def _deduplicate_sections(sections: list[dict]) -> list[dict]:
    """
    Deduplicate sections by section_id while preserving order.

    Parameters:
        sections: List of section dictionaries.

    Returns:
        Unique section dictionaries in first-seen order.
    """
    seen: set[str] = set()
    unique_sections: list[dict] = []

    for section in sections:
        section_id = _section_id(section)
        if not section_id or section_id in seen:
            continue

        seen.add(section_id)
        unique_sections.append(section)

    return unique_sections


def _filter_cites_edges(edges: list[dict], valid_section_ids: set[str]) -> list[dict]:
    """
    Keep only CITES edges whose source and target are present in the subgraph.

    Parameters:
        edges: Raw edge dictionaries returned by graph queries.
        valid_section_ids: Section IDs present in all_sections.

    Returns:
        Filtered CITES edges without dangling endpoints.
    """
    filtered_edges: list[dict] = []

    for edge in edges:
        source_id = edge.get("source_section_id") or edge.get("source_id")
        target_id = edge.get("target_section_id") or edge.get("target_id")
        edge_type = edge.get("type") or edge.get("relationship_type")

        if edge_type and edge_type != "CITES":
            continue

        if source_id in valid_section_ids and target_id in valid_section_ids:
            filtered_edges.append(edge)

    return filtered_edges


def traverse(
    concept_name: str,
    max_hops: int = 2,
    max_sections: int = 15,
    exact: bool = False,
) -> TraversalResult:
    """
    Traverse the legal knowledge graph for one plain-language legal concept.

    Parameters:
        concept_name: Plain-language legal concept, such as "wrongful termination".
        max_hops: Maximum CITES expansion depth. Defaults to 2.

    Returns:
        TraversalResult containing anchors, expanded sections, CITES edges,
        matched concepts, hop count, confidence, and empty-result status.
    """
    concept_name = concept_name.strip()

    primary_sections, supporting_sections, concepts_matched = _find_anchors(
        concept_name, exact=exact
    )

    confidence = _score_confidence(
        primary_count=len(primary_sections),
        supporting_count=len(supporting_sections),
    )

    if not primary_sections and not supporting_sections:
        logger.info(
            "Traversal completed for concept='%s'. No anchors found.",
            concept_name,
        )
        return TraversalResult(
            concept_queried=concept_name,
            anchor_sections=[],
            expanded_sections=[],
            all_sections=[],
            cites_edges=[],
            concepts_matched=[],
            hops_taken=0,
            confidence=0.0,
            is_empty=True,
        )

    primary_anchor_ids = [
        section_id
        for section in primary_sections
        if (section_id := _section_id(section))
    ]

    expanded_sections, hops_taken = _expand_from_anchors(
        anchor_section_ids=primary_anchor_ids,
        max_hops=max_hops,
    )

    expanded_sections = sorted(
        _deduplicate_sections(expanded_sections),
        key=_sort_section_key,
    )

    anchor_sections = _deduplicate_sections(primary_sections + supporting_sections)
    all_sections_seed = _deduplicate_sections(anchor_sections + expanded_sections)

    section_ids = [
        section_id
        for section in all_sections_seed
        if (section_id := _section_id(section))
    ]

    subgraph = get_subgraph(section_ids) or {}
    subgraph_sections = subgraph.get("sections") or all_sections_seed
    raw_edges = subgraph.get("cites_edges") or subgraph.get("edges") or []

    subgraph_section_by_id = {
        section_id: section
        for section in subgraph_sections
        if (section_id := _section_id(section))
    }

    all_sections: list[dict] = []
    for section in all_sections_seed:
        section_id = _section_id(section)
        all_sections.append(subgraph_section_by_id.get(section_id, section))

    all_sections = _deduplicate_sections(all_sections)

    valid_section_ids = {
        section_id
        for section in all_sections
        if (section_id := _section_id(section))
    }

    # Cap total sections passed to agent to prevent context overflow
    if len(all_sections) > max_sections:
        # keep all anchors, trim expanded sections
        all_sections = anchor_sections[:max_sections] + [
            s for s in expanded_sections
            if s not in anchor_sections
        ][:max(0, max_sections - len(anchor_sections))]

        all_sections = _deduplicate_sections(all_sections)

        # IMPORTANT:
        # recompute valid ids after trimming
        valid_section_ids = {
            section_id
            for section in all_sections
            if (section_id := _section_id(section))
        }

    cites_edges = _filter_cites_edges(raw_edges, valid_section_ids)

    logger.info(
        "Traversal completed for concept='%s'. anchors=%s expanded=%s total=%s edges=%s confidence=%s",
        concept_name,
        len(anchor_sections),
        len(expanded_sections),
        len(all_sections),
        len(cites_edges),
        confidence,
    )

    return TraversalResult(
        concept_queried=concept_name,
        anchor_sections=anchor_sections,
        expanded_sections=expanded_sections,
        all_sections=all_sections,
        cites_edges=cites_edges,
        concepts_matched=concepts_matched,
        hops_taken=hops_taken,
        confidence=confidence,
        is_empty=False,
    )


if __name__ == "__main__":
    import json

    test_concepts = [
        "wrongful termination",
        "gratuity",
        "minimum wages",
        "salary not paid",
        "unknown concept xyz",
    ]

    for concept in test_concepts:
        result = traverse(concept)

        print(f"\nConcept: '{concept}'")
        print(f"  Anchors:    {len(result.anchor_sections)}")
        print(f"  Expanded:   {len(result.expanded_sections)}")
        print(f"  Total:      {len(result.all_sections)}")
        print(f"  Edges:      {len(result.cites_edges)}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Empty:      {result.is_empty}")