from __future__ import annotations

from dataclasses import dataclass, field
import logging

from graph import act_registry
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
    # act_id -> why its sections were withheld (repealed / wrong state).
    suppressed_acts: dict[str, str] = field(default_factory=dict)


def _is_section(node: dict) -> bool:
    """
    True when a neighbour row is a Section.

    get_neighbors returns every adjacent node, which includes the Act via
    HAS_SECTION and Concept nodes via APPLIES_TO. Those carry no section_id, so
    checking for one identifies Sections without relying on labels (which the
    neighbour query does not return).
    """
    return bool(_section_id(node))


def _suppression_reason(
    section: dict,
    jurisdiction: str | None,
    drop_repealed: bool,
    filter_jurisdiction: bool,
) -> str | None:
    """
    Return why this section must be withheld, or None to keep it.

    Two independent filters, both deterministic and both applied BEFORE the
    section can reach the evidence pack:

      TEMPORAL     the owning Act has been repealed. Citing it would present
                   superseded law as operative.
      TERRITORIAL  the owning Act is state legislation and the user named a
                   DIFFERENT state. Karnataka's Shops Act does not govern an
                   employee in Maharashtra.

    A user who named NO state keeps state law (dropping it would make weekly
    holiday / annual leave unanswerable) — it is deprioritised and flagged
    instead, in agents/nodes/retrieval_node.py.
    """
    act_id = str(section.get("act_id") or "")

    if drop_repealed and str(section.get("in_force_status") or "in_force") == "repealed":
        note = act_registry.repeal_note(act_id)
        return note or f"The {act_id} has been repealed."

    if filter_jurisdiction and jurisdiction:
        act_jurisdiction = str(section.get("jurisdiction") or act_registry.CENTRAL)
        if (
            act_jurisdiction != act_registry.CENTRAL
            and act_jurisdiction != jurisdiction
        ):
            act_name = str(section.get("act_name") or act_id)
            return (
                f"The {act_name} applies only in {act_jurisdiction}, "
                f"and this query concerns {jurisdiction}."
            )

    return None


def _section_id(section: dict) -> str | None:
    return section.get("section_id") or section.get("id")


def _sort_section_key(section: dict) -> tuple[str, str]:
    return (
        str(section.get("act_id") or ""),
        str(section.get("section_number") or ""),
    )


def _find_anchors(
    concept_name: str,
    exact: bool = False,
    jurisdiction: str | None = None,
    drop_repealed: bool = True,
    filter_jurisdiction: bool = True,
) -> tuple[list[dict], list[dict], list[str], dict[str, str]]:
    """
    Find primary and supporting section anchors for a concept.

    Parameters:
        concept_name: Plain-language legal concept extracted by the agent layer.
        exact: When True, match the concept by exact canonical name
               (agent layer already grounded it). When False, use the
               looser CONTAINS matching (main.py / raw-text callers).
        jurisdiction: Resolved jurisdiction of the query ("Karnataka",
               "Central", ...) or None when the user named no state.
        drop_repealed: Withhold sections whose Act has been repealed.
        filter_jurisdiction: Withhold state legislation from another state.

    Returns:
        A tuple containing:
        - primary_sections: Section nodes marked as primary anchors.
        - supporting_sections: Section nodes marked as supporting anchors.
        - concepts_matched: Concept names matched in the graph.
        - suppressed_acts: act_id -> reason, for anchors that were withheld.

    Filtering happens HERE, before confidence scoring and before BFS, and that
    placement is load-bearing. Filtering later would let a repealed section act
    as a primary anchor: it would score traversal confidence 1.0 and seed the
    CITES expansion, so suppressed law would still steer which live sections got
    retrieved.
    """
    rows = (
        get_sections_for_exact_concept(concept_name)
        if exact
        else get_sections_for_concept(concept_name)
    ) or []

    primary_sections: list[dict] = []
    supporting_sections: list[dict] = []
    concepts_matched: list[str] = []
    suppressed_acts: dict[str, str] = {}

    for row in rows:
        section = row.get("section", row)

        reason = _suppression_reason(
            section, jurisdiction, drop_repealed, filter_jurisdiction
        )
        if reason:
            act_id = str(section.get("act_id") or "")
            suppressed_acts.setdefault(act_id, reason)
            continue

        # NOTE: "relevance" is the key graph/queries.py actually returns (it is
        # the APPLIES_TO edge property). It was missing from this chain, so every
        # anchor fell through to `primary` — which made _score_confidence report
        # 1.0 whenever anything matched at all, and seeded the CITES expansion
        # from supporting sections as though they were operative provisions.
        relation_type = (
            row.get("relevance")
            or row.get("match_type")
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

    return primary_sections, supporting_sections, concepts_matched, suppressed_acts


def _expand_from_anchors(
    anchor_section_ids: list[str],
    max_hops: int,
    jurisdiction: str | None = None,
    drop_repealed: bool = True,
    filter_jurisdiction: bool = True,
) -> tuple[list[dict], int, dict[str, str]]:
    """
    Expand from primary anchor sections using BFS over CITES relationships.

    Parameters:
        anchor_section_ids: Section IDs used as traversal starting points.
        max_hops: Maximum number of expansion rounds.
        jurisdiction / drop_repealed / filter_jurisdiction: same meaning as in
            _find_anchors — suppressed sections are neither returned NOR walked
            through, so repealed law cannot act as a bridge to other sections.

    Returns:
        A tuple containing:
        - expanded_sections: Newly discovered Section nodes, excluding anchors.
          Each carries a `hop_distance` key recording how far out it was found.
        - hops_taken: Number of hop rounds actually performed.
        - suppressed_acts: act_id -> reason, for neighbours that were withheld.
    """
    if max_hops <= 0 or not anchor_section_ids:
        return [], 0, {}

    visited = set(anchor_section_ids)
    frontier = set(anchor_section_ids)
    expanded_sections: list[dict] = []
    suppressed_acts: dict[str, str] = {}
    hops_taken = 0

    for hop in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        new_sections_this_hop: list[dict] = []

        for section_id in frontier:
            neighbors = get_neighbors(section_id) or []

            for neighbor in neighbors:
                # Expansion is defined over CITES (section cross-references).
                # get_neighbors also returns the owning Act via HAS_SECTION and
                # Concept nodes via APPLIES_TO; walking those would jump to an
                # unrelated part of the graph.
                if str(neighbor.get("rel_type") or "") != "CITES":
                    continue

                neighbor_section = neighbor.get("node", neighbor)
                if not _is_section(neighbor_section):
                    continue

                neighbor_id = _section_id(neighbor_section)
                if not neighbor_id or neighbor_id in visited:
                    continue

                visited.add(neighbor_id)

                reason = _suppression_reason(
                    neighbor_section, jurisdiction, drop_repealed, filter_jurisdiction
                )
                if reason:
                    suppressed_acts.setdefault(
                        str(neighbor_section.get("act_id") or ""), reason
                    )
                    continue

                enriched = dict(neighbor_section)
                enriched["hop_distance"] = hop
                next_frontier.add(neighbor_id)
                new_sections_this_hop.append(enriched)

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

    return expanded_sections, hops_taken, suppressed_acts


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
    jurisdiction: str | None = None,
    drop_repealed: bool = True,
    filter_jurisdiction: bool = True,
) -> TraversalResult:
    """
    Traverse the legal knowledge graph for one plain-language legal concept.

    Parameters:
        concept_name: Plain-language legal concept, such as "wrongful termination".
        max_hops: Maximum CITES expansion depth. Defaults to 2.
        max_sections: Cap on the returned section count.
        exact: Match the concept by exact canonical name (agent layer) rather
            than the looser CONTAINS matching (main.py / raw-text callers).
        jurisdiction: Resolved jurisdiction of the query, or None if the user
            named no state. State legislation from a DIFFERENT state is withheld.
        drop_repealed: Withhold sections belonging to a repealed Act.
        filter_jurisdiction: Enable the territorial filter.

    Returns:
        TraversalResult containing anchors, expanded sections, CITES edges,
        matched concepts, hop count, confidence, empty-result status, and the
        acts that were withheld with the reason for each.
    """
    concept_name = concept_name.strip()

    primary_sections, supporting_sections, concepts_matched, suppressed_acts = (
        _find_anchors(
            concept_name,
            exact=exact,
            jurisdiction=jurisdiction,
            drop_repealed=drop_repealed,
            filter_jurisdiction=filter_jurisdiction,
        )
    )

    confidence = _score_confidence(
        primary_count=len(primary_sections),
        supporting_count=len(supporting_sections),
    )

    if not primary_sections and not supporting_sections:
        logger.info(
            "Traversal completed for concept='%s'. No anchors found%s",
            concept_name,
            f" ({len(suppressed_acts)} act(s) suppressed)." if suppressed_acts else ".",
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
            suppressed_acts=suppressed_acts,
        )

    # Anchors are hop 0 by definition — they were reached through APPLIES_TO,
    # not by following a citation.
    for section in primary_sections + supporting_sections:
        section["hop_distance"] = 0

    primary_anchor_ids = [
        section_id
        for section in primary_sections
        if (section_id := _section_id(section))
    ]

    expanded_sections, hops_taken, expansion_suppressed = _expand_from_anchors(
        anchor_section_ids=primary_anchor_ids,
        max_hops=max_hops,
        jurisdiction=jurisdiction,
        drop_repealed=drop_repealed,
        filter_jurisdiction=filter_jurisdiction,
    )
    for act_id, reason in expansion_suppressed.items():
        suppressed_acts.setdefault(act_id, reason)

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

    # Merge in the richer subgraph row, but carry the traversal provenance
    # across: the subgraph query knows nothing about how far out we walked, and
    # losing hop_distance here would blind the ranker (graph/ranking.py) to the
    # difference between an anchor and a section three citations away.
    all_sections: list[dict] = []
    for section in all_sections_seed:
        section_id = _section_id(section)
        merged = dict(subgraph_section_by_id.get(section_id, section))
        merged["hop_distance"] = int(section.get("hop_distance", 0) or 0)
        all_sections.append(merged)

    all_sections = _deduplicate_sections(all_sections)

    # Safety valve on context size. The REAL relevance ordering happens in
    # agents/nodes/retrieval_node.py, which ranks the union across every
    # grounded concept against the user's actual query text — something this
    # function cannot do, since it only ever sees one concept name. So the trim
    # here just has to be principled and lossless where it matters: keep every
    # anchor, then take the nearest hops.
    if len(all_sections) > max_sections:
        anchor_ids = {
            section_id
            for section in anchor_sections
            if (section_id := _section_id(section))
        }
        anchors_first = [s for s in all_sections if _section_id(s) in anchor_ids]
        expanded_rest = sorted(
            (s for s in all_sections if _section_id(s) not in anchor_ids),
            key=lambda s: (int(s.get("hop_distance", 0) or 0), _sort_section_key(s)),
        )
        remaining = max(0, max_sections - len(anchors_first))
        all_sections = _deduplicate_sections(
            anchors_first[:max_sections] + expanded_rest[:remaining]
        )

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
        suppressed_acts=suppressed_acts,
    )


if __name__ == "__main__":

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