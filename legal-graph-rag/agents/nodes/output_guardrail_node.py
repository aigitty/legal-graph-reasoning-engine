"""
agents/nodes/output_guardrail_node.py

Deterministic OUTPUT GUARDRAIL — the integrity-enforcement stage of the
pipeline (CLAUDE.md rules 3 & 5). Runs AFTER answer_synthesis and BEFORE
final_response. No LLM call, no graph mutation (READ-ONLY against Neo4j).

This node is where the project's core guarantee is ENFORCED (synthesis only
ASKS for it via the prompt). It does four deterministic things:

  1. VERIFY every citation produced by synthesis. A cited section id survives
     only if it passes BOTH checks:
        a. provenance — it was in the retrieval set actually shown to the LLM
           (state.retrieval.section_ids), AND
        b. existence  — it really is a Section node in Neo4j
           (graph.queries.get_section_by_id).
     Any citation failing either check is STRIPPED and logged as a warning.

  2. COMPUTE the confidence score deterministically (CLAUDE.md section 6):
        confidence = 0.35 * concept_coverage
                   + 0.25 * seed_strength
                   + 0.20 * sufficiency_score
                   + 0.20 * citation_validity
     Each factor is a value in [0, 1]; the weights sum to 1.0, so confidence
     is in [0, 1].

  3. SET status — "ok", downgraded to "insufficient_evidence" when confidence
     falls below MIN_CONFIDENCE.

  4. INJECT the legal disclaimer in code (never left to the LLM).

Degrade-honestly (CLAUDE.md section 7): if Neo4j is unreachable we fall back to
provenance-only verification — sections in the retrieval set were already
fetched from Neo4j during traversal, so they are known-good — and record a
warning, rather than blocking the whole answer.
"""

from __future__ import annotations

import logging

from agents.calculators import Calculation
from agents.graph_state import LegalQueryState
from graph.queries import get_section_by_id
from config import cfg

logger = logging.getLogger(__name__)

# CLAUDE.md section 6 — deterministic confidence weights (must sum to 1.0).
# Values come from config so they can be tuned without touching node code.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "concept_coverage": cfg.CONFIDENCE_W_CONCEPT,
    "seed_strength": cfg.CONFIDENCE_W_SEED,
    "sufficiency_score": cfg.CONFIDENCE_W_SUFFICIENCY,
    "citation_validity": cfg.CONFIDENCE_W_CITATION,
}

MIN_CONFIDENCE = cfg.MIN_CONFIDENCE

# Injected deterministically. The synthesis prompt is explicitly told NOT to
# emit this so it never appears twice and is never the LLM's to phrase.
DISCLAIMER = (
    "This is general legal information generated from retrieved statutory text, "
    "not legal advice. It may be incomplete and may not reflect the latest "
    "amendments or judicial interpretation. For guidance on your specific "
    "situation, consult a qualified lawyer or the appropriate labour authority."
)


# ---------------------------------------------------------------------------
# Citation verification (the integrity gate)
# ---------------------------------------------------------------------------

def _verify_citations(state: LegalQueryState) -> tuple[list[str], list[str]]:
    """
    Return (verified_ids, warnings).

    A citation is kept only if it was in the retrieval set AND exists in Neo4j.
    Order is preserved (first-cited first). If Neo4j becomes unreachable we stop
    hitting it and fall back to provenance-only verification for the remainder.
    """
    cited = state.cited_section_ids or []
    retrieval_ids = state.retrieval.section_ids if state.retrieval is not None else set()

    verified: list[str] = []
    warnings: list[str] = []
    db_available = True

    for cid in cited:
        # (a) provenance — the LLM may only cite what it was shown.
        if cid not in retrieval_ids:
            warnings.append(
                f"Stripped citation {cid!r}: not in the retrieval set "
                f"(model cited a section it was not shown)."
            )
            continue

        # (b) existence — confirm against Neo4j while the DB is reachable.
        if db_available:
            try:
                if get_section_by_id(cid) is None:
                    warnings.append(
                        f"Stripped citation {cid!r}: not found as a Section node "
                        f"in Neo4j."
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                db_available = False
                logger.warning(
                    "Neo4j citation verification unavailable (%s); falling back "
                    "to provenance-only checks for remaining citations.",
                    exc,
                )
                warnings.append(
                    "Live Neo4j citation verification was unavailable; remaining "
                    "citations were verified against the retrieval set only."
                )

        verified.append(cid)

    return verified, warnings


# ---------------------------------------------------------------------------
# Confidence factors — each returns a value in [0, 1]
# ---------------------------------------------------------------------------

def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _concept_coverage(state: LegalQueryState) -> float:
    """
    How much of the query's legal intent was grounded to known concepts.

    With extracted concepts present: the fraction of EXTRACTED PHRASES that
    grounded to at least one canonical concept. Without extraction: 1.0 if
    anything grounded, else 0.0.

    This used to be computed as len(grounded_concepts) / len(extracted), which
    measured the wrong thing in both directions. Grounding unions its matches
    across phrases, so one phrase hitting three concepts made the ratio exceed
    1.0 (silently clamped) and masked a second phrase that grounded to nothing;
    conversely two phrases collapsing onto the same concept scored 0.5 despite
    full coverage. Counting ungrounded phrases measures coverage directly.
    """
    grounded = state.grounded_concepts or []

    if state.extraction is not None and state.extraction.legal_concepts:
        extracted = state.extraction.legal_concepts
        ungrounded = len(state.ungrounded_phrases)
        return _clamp((len(extracted) - ungrounded) / len(extracted))

    return 1.0 if grounded else 0.0


def _seed_strength(state: LegalQueryState) -> float:
    """
    Strength of the retrieval anchors. Reuses the traversal confidence
    (1.0 with primary anchors, 0.6 with only supporting anchors, 0.0 none).
    """
    if state.retrieval is None:
        return 0.0
    return _clamp(float(state.retrieval.confidence))


def _sufficiency_score(state: LegalQueryState) -> float:
    """
    From the sufficiency verdict: 1.0 if sufficient; 0.5 if insufficient but
    some sections were still retrieved (an honest partial answer is possible);
    0.0 if no verdict or nothing retrieved.
    """
    if state.sufficiency is None:
        return 0.0
    if state.sufficiency.sufficient:
        return 1.0
    if state.retrieval is not None and not state.retrieval.is_empty:
        return 0.5
    return 0.0


def _verify_calculations(
    state: LegalQueryState, retrieval_ids: set[str]
) -> tuple[list[Calculation], list[str]]:
    """
    Keep only Calculations whose section exists in Neo4j.

    Provenance (the section was actually retrieved for this query) is already
    guaranteed at the point calculators.compute_available() is called in
    retrieval_node — it is gated on the same retrieved_section_ids the ranker
    produced. This adds the EXISTENCE half of the same two-part check ordinary
    citations get (CLAUDE.md rule 5): a Calculation is still a claim that cites
    a section, and gets no exemption from verification for being arithmetic
    instead of prose. Degrades the same way _verify_citations does if Neo4j
    becomes unreachable — provenance-only, with a warning.
    """
    verified: list[Calculation] = []
    warnings: list[str] = []
    db_available = True

    for calc in state.calculations or []:
        if calc.section_id not in retrieval_ids:
            warnings.append(
                f"Dropped calculation {calc.label!r}: its section "
                f"{calc.section_id!r} was not in the retrieval set."
            )
            continue

        if db_available:
            try:
                if get_section_by_id(calc.section_id) is None:
                    warnings.append(
                        f"Dropped calculation {calc.label!r}: section "
                        f"{calc.section_id!r} not found in Neo4j."
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                db_available = False
                logger.warning(
                    "Neo4j calculation verification unavailable (%s); "
                    "falling back to provenance-only checks.",
                    exc,
                )

        verified.append(calc)

    return verified, warnings


def _citation_validity(verified: list[str], cited: list[str]) -> float:
    """Fraction of the model's citations that survived verification."""
    if not cited:
        return 0.0
    return _clamp(len(verified) / len(cited))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def output_guardrail_node(state: LegalQueryState) -> dict:
    verified, citation_warnings = _verify_citations(state)
    cited = state.cited_section_ids or []

    retrieval_ids = state.retrieval.section_ids if state.retrieval is not None else set()
    verified_calculations, calc_warnings = _verify_calculations(state, retrieval_ids)

    factors = {
        "concept_coverage": round(_concept_coverage(state), 4),
        "seed_strength": round(_seed_strength(state), 4),
        "sufficiency_score": round(_sufficiency_score(state), 4),
        "citation_validity": round(_citation_validity(verified, cited), 4),
    }
    confidence = round(
        sum(CONFIDENCE_WEIGHTS[name] * value for name, value in factors.items()),
        4,
    )

    warnings = list(citation_warnings) + list(calc_warnings)

    # A temporal mismatch is not a grounding, retrieval, or citation problem —
    # every factor above can legitimately score 1.0 — but it means the cited
    # law's APPLICABILITY to these specific facts is genuinely uncertain, which
    # the numeric score must reflect. Capped, not zeroed: the sections are still
    # real, verified, and often close to right (see CLAUDE.md — IDA s.25F and
    # IRC s.70 both give one month's notice).
    if state.temporal_conflicts:
        capped = min(confidence, cfg.TEMPORAL_MISMATCH_CONFIDENCE_CAP)
        if capped < confidence:
            warnings.append(
                f"Confidence capped at {capped:.2f}: the query's event predates "
                f"the commencement of the cited law, so its applicability to "
                f"these facts is uncertain."
            )
        confidence = capped

    if confidence < MIN_CONFIDENCE:
        status = "insufficient_evidence"
        warnings.append(
            f"Confidence {confidence:.2f} is below the {MIN_CONFIDENCE} threshold; "
            f"status downgraded to insufficient_evidence."
        )
    else:
        status = "ok"

    logger.info(
        "Output guardrail: %s/%s citations verified, confidence=%.2f, status=%s",
        len(verified),
        len(cited),
        confidence,
        status,
    )

    return {
        "verified_section_ids": verified,
        "verified_calculations": verified_calculations,
        "confidence": confidence,
        "confidence_factors": factors,
        "status": status,
        "disclaimer": DISCLAIMER,
        "warnings": state.warnings + warnings,
    }
