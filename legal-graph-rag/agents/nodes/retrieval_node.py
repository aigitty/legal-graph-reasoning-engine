"""
agents/nodes/retrieval_node.py

Deterministic graph retrieval node.

Loops over state.grounded_concepts, calls graph.traversal.traverse() with
exact=True (canonical concept names from grounding) once per concept, and merges
the results: dedup sections by section_id, relevance from
agents.ontology.relevance_for(), dedup CITES edges, union acts_covered.

max_hops comes from state (default 2, mutated by graph_expansion_node on the
sufficiency loop). Still fully DETERMINISTIC: no LLM, no invented sections.

THREE THINGS HAPPEN HERE THAT DID NOT BEFORE
--------------------------------------------
1. TEMPORAL FILTERING. Sections belonging to a repealed Act are withheld.
   The graph previously returned the Minimum Wages Act 1948 and the Code on
   Wages 2019 side by side, and the answer presented both as simultaneously
   operative even though COW s.69 expressly repealed the MWA.

2. TERRITORIAL FILTERING. `jurisdiction` was extracted by the LLM, threaded
   through the whole pipeline, and then ignored — `jurisdiction_applied` was
   hardcoded to "Central". A user in Maharashtra asking about shop working
   hours was answered with the KARNATAKA Shops Act. State legislation is now
   withheld when the user named a different state, and deprioritised (with a
   warning) when they named none.

3. RELEVANCE RANKING. The union across concepts is scored by graph/ranking.py
   against the user's actual query and cut to cfg.MAX_SECTIONS. Previously the
   cap sliced (act_id, section_number) order, so which sections survived was
   effectively alphabetical.

Suppression is never silent: every withheld Act lands in
RetrievalResult.suppressed_acts with a reason, and a warning is raised so the
answer can say what was withheld and why.
"""

from __future__ import annotations

import logging
from typing import Any

from graph import act_registry
from graph.queries import get_section_by_id
from graph.ranking import RankCandidate, rank_and_cap
from graph.traversal import traverse
from agents.state import RetrievalResult, SectionContext
from agents.graph_state import LegalQueryState
from agents.ontology import relevance_for
from config import cfg

logger = logging.getLogger(__name__)

MAX_SECTIONS = cfg.MAX_SECTIONS


def _normalize_edge(edge: dict[str, Any]) -> dict[str, str]:
    source = edge.get("source_section_id") or edge.get("source_id") or ""
    target = edge.get("target_section_id") or edge.get("target_id") or ""
    return {"source_section_id": str(source), "target_section_id": str(target)}


def _relevance_rank(relevance: str) -> int:
    return 1 if relevance == "primary" else 0


def _resolve_jurisdiction(state: LegalQueryState) -> str | None:
    """
    Resolve the jurisdiction this query should be answered under.

    Returns the canonical jurisdiction ("Karnataka", "Maharashtra", "Central"),
    or None when the user named no state at all — a distinction that matters,
    because a named-but-different state means "withhold that state's law" while
    an unnamed state means "keep it, rank it lower, and say so".
    """
    if state.extraction is None:
        return None
    return act_registry.normalize_jurisdiction(state.extraction.jurisdiction)


def _direct_section_lookups(
    state: LegalQueryState, jurisdiction: str | None
) -> list[SectionContext]:
    """
    Fetch sections the user named explicitly ("what does Section 25N say?").

    Concept grounding cannot serve a lookup question: "Section 25N of the
    Industrial Disputes Act" grounds to `industrial dispute`, which retrieves
    the dispute-machinery provisions and never returns 25N itself. This is a
    separate, deterministic path — a direct id fetch, verified against Neo4j,
    subject to the same temporal and territorial filters as everything else.

    Returned as hop 0 PRIMARY: the user pointed straight at it, so nothing in
    the pack should outrank it.
    """
    candidates = act_registry.resolve_section_references(state.raw_query)
    if not candidates:
        return []

    found: list[SectionContext] = []
    for section_id in candidates:
        try:
            row = get_section_by_id(section_id)
        except Exception as exc:  # noqa: BLE001 — degrade, never block
            logger.warning("Direct section lookup failed for %r: %s", section_id, exc)
            break
        if not row:
            continue

        act_id = act_registry.act_id_for_section(section_id)
        meta = act_registry.get_act(act_id)
        if meta is not None:
            if cfg.SUPPRESS_REPEALED_ACTS and not meta.is_in_force:
                continue
            if (
                cfg.FILTER_BY_JURISDICTION
                and jurisdiction
                and not meta.is_central
                and meta.jurisdiction != jurisdiction
            ):
                continue

        data = dict(row)
        data.setdefault("section_id", section_id)
        data["act_id"] = act_id
        if meta is not None:
            data["act_name"] = meta.act_name
            data["jurisdiction"] = meta.jurisdiction
            data["in_force_status"] = meta.status
            data["act_priority"] = meta.act_priority

        found.append(
            SectionContext.from_dict(
                data=data,
                relevance="primary",
                source_concept="direct section reference",
                hop_distance=0,
            )
        )

    if found:
        logger.info(
            "Direct section lookup matched: %s", [s.section_id for s in found]
        )
    return found


def graph_traversal_node(state: LegalQueryState) -> dict[str, Any]:
    concepts = state.grounded_concepts or (
        [state.raw_query.strip().lower()] if state.raw_query.strip() else []
    )

    if not concepts:
        return {"error": "No grounded concepts and no raw_query to fall back to."}

    jurisdiction = _resolve_jurisdiction(state)
    relevance_map = relevance_for(concepts)

    sections_by_id: dict[str, SectionContext] = {}
    concept_hits: dict[str, int] = {}
    edges_seen: dict[tuple[str, str], dict[str, str]] = {}
    acts_covered: list[str] = []
    acts_seen: set[str] = set()
    suppressed_acts: dict[str, str] = {}
    confidences: list[float] = []

    for concept in concepts:
        try:
            tr = traverse(
                concept_name=concept,
                max_hops=state.max_hops,
                max_sections=cfg.MAX_SECTIONS_PER_CONCEPT,
                exact=True,
                jurisdiction=jurisdiction,
                drop_repealed=cfg.SUPPRESS_REPEALED_ACTS,
                filter_jurisdiction=cfg.FILTER_BY_JURISDICTION,
            )
        except Exception as exc:
            logger.exception("Graph traversal failed for concept %r", concept)
            return {"error": f"Graph traversal failed for {concept!r}: {exc}"}

        for act_id, reason in (getattr(tr, "suppressed_acts", None) or {}).items():
            suppressed_acts.setdefault(act_id, reason)

        confidences.append(float(getattr(tr, "confidence", 0.0) or 0.0))

        for raw_section in (getattr(tr, "all_sections", []) or []):
            section_id = str(raw_section.get("section_id") or raw_section.get("id") or "")
            sec = SectionContext.from_dict(
                data=raw_section,
                relevance=relevance_map.get(section_id, "supporting"),
                source_concept=concept,
                hop_distance=int(raw_section.get("hop_distance", 0) or 0),
            )

            existing = sections_by_id.get(sec.section_id)
            if existing is None:
                sections_by_id[sec.section_id] = sec
                concept_hits[sec.section_id] = 1
                if sec.act_id and sec.act_id not in acts_seen:
                    acts_seen.add(sec.act_id)
                    acts_covered.append(sec.act_id)
            else:
                concept_hits[sec.section_id] = concept_hits.get(sec.section_id, 1) + 1
                if _relevance_rank(sec.relevance) > _relevance_rank(existing.relevance):
                    existing.relevance = sec.relevance
                # Keep the SHORTEST route to this section: a section that is an
                # anchor for one concept should not be scored as a distant hop
                # just because another concept reached it the long way round.
                existing.hop_distance = min(existing.hop_distance, sec.hop_distance)
                if concept not in existing.source_concept.split(", "):
                    existing.source_concept = f"{existing.source_concept}, {concept}"

        for raw_edge in (getattr(tr, "cites_edges", None) or []):
            edge = _normalize_edge(raw_edge)
            edges_seen[(edge["source_section_id"], edge["target_section_id"])] = edge

    warnings: list[str] = []

    # ---- sections the user named explicitly --------------------------------
    for section in _direct_section_lookups(state, jurisdiction):
        existing = sections_by_id.get(section.section_id)
        if existing is None:
            sections_by_id[section.section_id] = section
            concept_hits[section.section_id] = 1
            if section.act_id and section.act_id not in acts_seen:
                acts_seen.add(section.act_id)
                acts_covered.append(section.act_id)
        else:
            # Already reached by traversal — promote it, since an explicit
            # reference is the strongest possible relevance signal.
            existing.relevance = "primary"
            existing.hop_distance = 0

    # ---- rank the union against the actual query, then cap -----------------
    sections = list(sections_by_id.values())
    for section in sections:
        section.concept_hits = concept_hits.get(section.section_id, 1)

    candidates = [
        RankCandidate(
            section_id=s.section_id,
            title=s.section_title,
            text=s.section_text,
            relevance=s.relevance,
            hop_distance=s.hop_distance,
            concept_hits=s.concept_hits,
            act_priority=s.act_priority,
            raw=s,
        )
        for s in sections
    ]

    ranked = rank_and_cap(state.raw_query, candidates, limit=MAX_SECTIONS)

    ordered_sections: list[SectionContext] = []
    for candidate in ranked:
        section: SectionContext = candidate.raw  # type: ignore[assignment]
        score = candidate.score
        # State law kept only because the user named no state ranks below
        # Central law, and the user is told the assumption was made.
        if (
            jurisdiction is None
            and section.jurisdiction != act_registry.CENTRAL
        ):
            score *= cfg.UNSTATED_JURISDICTION_PENALTY
        section.score = round(score, 6)
        ordered_sections.append(section)

    ordered_sections.sort(key=lambda s: -s.score)

    kept_ids = {s.section_id for s in ordered_sections}
    cites_edges = [
        edge
        for edge in edges_seen.values()
        if edge["source_section_id"] in kept_ids
        and edge["target_section_id"] in kept_ids
    ]
    acts_covered = [
        act_id for act_id in acts_covered
        if any(s.act_id == act_id for s in ordered_sections)
    ]

    # ---- surface every suppression and assumption --------------------------
    for act_id, reason in suppressed_acts.items():
        warnings.append(f"Withheld {act_id}: {reason}")

    state_acts_kept = sorted(
        {
            s.jurisdiction
            for s in ordered_sections
            if s.jurisdiction != act_registry.CENTRAL
        }
    )
    if jurisdiction is None and state_acts_kept:
        warnings.append(
            "No state was specified in the query, so state legislation was kept "
            "but ranked below central law. The following applies only in "
            + ", ".join(state_acts_kept)
            + "; confirm the state before relying on it."
        )

    result = RetrievalResult(
        concepts_queried=concepts,
        sections=ordered_sections,
        cites_edges=cites_edges,
        total_found=len(ordered_sections),
        acts_covered=acts_covered,
        confidence=max(confidences) if confidences else 0.0,
        is_empty=not ordered_sections,
        jurisdiction_applied=jurisdiction or act_registry.CENTRAL,
        suppressed_acts=suppressed_acts,
    )

    logger.info(
        "Retrieval: %s concepts -> %s sections (jurisdiction=%s, suppressed=%s)",
        len(concepts),
        len(ordered_sections),
        result.jurisdiction_applied,
        list(suppressed_acts),
    )

    update: dict[str, Any] = {"retrieval": result}
    if warnings:
        update["warnings"] = state.warnings + warnings
    return update
