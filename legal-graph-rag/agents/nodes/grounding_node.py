"""
agents/nodes/grounding_node.py

Deterministic concept-grounding node.

Grounds each phrase in state.extraction.legal_concepts against
data/ontology/concept_map.json via agents.ontology.ground_query(), unioning the
matches, and ALSO grounds state.raw_query itself. No LLM here.
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
    ungrounded: list[str] = []

    for text in candidates:
        matches = ground_query(text)
        # Track PER PHRASE, not just the union. The output guardrail scores
        # concept_coverage as "how much of the query's legal intent did we
        # actually ground", which is a question about the extracted phrases —
        # and one phrase can ground to several concepts while another grounds to
        # none. Counting only the union hides that second phrase entirely.
        if not matches:
            ungrounded.append(text)
            continue
        for concept_name in matches:
            if concept_name not in seen:
                seen.add(concept_name)
                grounded.append(concept_name)

    # ALSO ground the raw query, not only as a fallback when extraction is empty.
    #
    # Extraction paraphrases, and a paraphrase can drop the most important fact
    # in the question. Asked "I work for a food delivery app — do I get any
    # social security benefits?" it returned "social security benefits", which
    # grounds to ESI and maternity benefit but loses that the user is a GIG
    # WORKER — the one thing that determines which law applies to them. The
    # user's own words still contained it.
    #
    # These matches are deliberately NOT counted towards `ungrounded_phrases`:
    # concept_coverage measures how much of the EXTRACTED intent was grounded,
    # so a raw-query hit is a bonus, not evidence about extraction quality.
    for concept_name in ground_query(state.raw_query):
        if concept_name not in seen:
            seen.add(concept_name)
            grounded.append(concept_name)

    update: dict = {"grounded_concepts": grounded, "ungrounded_phrases": ungrounded}

    if not grounded:
        update["warnings"] = state.warnings + [
            f"No concept matched for candidates: {candidates!r}"
        ]
    elif ungrounded:
        update["warnings"] = state.warnings + [
            "No concept matched for part of the query: "
            + ", ".join(repr(phrase) for phrase in ungrounded)
            + ". That aspect may not be covered by the retrieved law."
        ]

    return update
