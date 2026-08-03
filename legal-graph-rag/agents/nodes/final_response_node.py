"""
agents/nodes/final_response_node.py

FINAL RESPONSE assembler — the terminal node for every exit path. Pure
formatting: NO LLM call, NO Neo4j call. It only reshapes state into the
single user-facing `final_answer` string.

This node is the pipeline's SAFETY NET, so it MUST NEVER raise. Every code
path is wrapped; on any unexpected failure it returns a fixed honest fallback
and marks status="error" rather than letting an exception escape into the
graph runtime.

It produces an answer for all five terminal statuses:
  - ok                  -> the synthesized answer + verified citations,
                           confidence, and disclaimer
  - insufficient_evidence -> the same, framed honestly as PARTIAL, naming the
                           sufficiency gap
  - out_of_domain       -> honest "outside scope" message, NO legal content
  - rejected            -> safety refusal, NO legal content
  - error               -> internal-error message, no legal conclusions

Status precedence (CLAUDE.md section 7 — safety / domain exits must win even
if synthesis happened to run upstream): error > rejected > out_of_domain >
whatever the output guardrail set (ok / insufficient_evidence).

Citation hygiene: the synthesized draft carries inline [SECTION_ID] markers.
Any marker NOT in verified_section_ids is stripped from the displayed text so
that an unverified citation can never reach the user (CLAUDE.md rule 5).

Persona rendering of markers (presentation only — verification has ALREADY
happened in output_guardrail_node, which runs before this node):
  - lawyer  -> verified markers are KEPT inline; a practitioner uses them.
  - citizen -> verified markers are also removed from the visible text, because
               "[COW_2019_S18]" mid-sentence is noise to a layperson who cannot
               do anything with a machine id. The law is still named in plain
               words by the synthesis prompt, and the trailer lists every
               verified section in human-readable form.
"""

from __future__ import annotations

import logging

from agents.citations import render_markers
from agents.graph_state import LegalQueryState
from agents.persona import CITIZEN, LAWYER, normalize_persona
from config import cfg
from graph import act_registry

logger = logging.getLogger(__name__)

_SAFE_FALLBACK = (
    "The system was unable to assemble a final answer for this query due to an "
    "internal error. No legal conclusions should be drawn from this response. "
    "Please try again."
)

# Terminal statuses that show a fixed honest message and NO legal content.
# Citations and confidence are cleared for these (see final_response_node).
_NO_CONTENT_STATUSES = frozenset({"error", "rejected", "out_of_domain"})

# DERIVED from act_metadata rather than hardcoded, so the scope we advertise
# always matches the scope we will actually cite from. act_registry reads a local
# JSON file — no Neo4j, no LLM — so this node stays pure formatting.
_SCOPE_LINE = act_registry.scope_sentence()


# ---------------------------------------------------------------------------
# Status resolution
# ---------------------------------------------------------------------------

def _resolve_status(state: LegalQueryState) -> str:
    """Apply CLAUDE.md section-7 precedence over the guardrail's status."""
    if state.error:
        return "error"
    extraction = state.extraction
    if extraction is not None:
        if extraction.safety_flag:
            return "rejected"
        if not extraction.in_domain:
            return "out_of_domain"
    return state.status or "ok"


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

# Marker rendering lives in agents/citations.py, shared with synthesis_node's
# parser, so the two can never disagree about what counts as a citation.
_render_markers = render_markers


def _confidence_word(score: float, partial: bool) -> str:
    # Plain-language band for the citizen trailer (thresholds mirror the
    # MIN_CONFIDENCE downgrade in the output guardrail).
    if score >= cfg.CITIZEN_CONFIDENCE_HIGH:
        word = "high"
    elif score >= cfg.CITIZEN_CONFIDENCE_MODERATE:
        word = "moderate"
    else:
        word = "low"
    # Never tell a citizen an answer is "high" confidence while the same answer
    # admits the retrieved law only partially covers their question — the two
    # statements read as a contradiction. The numeric score is unchanged; only
    # the word shown is capped.
    if partial and word == "high":
        word = "moderate"
    return word


def _evidence_is_partial(state: LegalQueryState, partial: bool) -> bool:
    """
    True when the answer itself admits a gap. Note this is BROADER than
    status == "insufficient_evidence": the loop can exhaust with
    sufficient=False and still score above MIN_CONFIDENCE, so status stays "ok"
    while the answer text names an unanswered part of the question.
    """
    if partial:
        return True
    return state.sufficiency is not None and not state.sufficiency.sufficient


def _confidence_line(state: LegalQueryState, persona: str, partial: bool) -> str:
    if persona == LAWYER:
        f = state.confidence_factors or {}
        if f:
            breakdown = (
                f"concept_coverage {f.get('concept_coverage', 0.0):.2f}"
                f" · seed_strength {f.get('seed_strength', 0.0):.2f}"
                f" · sufficiency {f.get('sufficiency_score', 0.0):.2f}"
                f" · citation_validity {f.get('citation_validity', 0.0):.2f}"
            )
            return f"Confidence: {state.confidence:.2f}  ({breakdown})"
        return f"Confidence: {state.confidence:.2f}"
    # Citizen: no numeric jargon or factor breakdown.
    word = _confidence_word(state.confidence, _evidence_is_partial(state, partial))
    return f"How confident is this answer: {word}."


def _display_names(state: LegalQueryState, verified: list[str]) -> list[str]:
    """
    Turn verified section ids into human-readable names using the retrieval set
    (e.g. "COW_2019_S18" -> "Section 18 of the Code on Wages, 2019"). Falls back
    to the raw id if the section is somehow not in the retrieval set — it is
    still a verified id, so it is never dropped.
    """
    lookup = {}
    if state.retrieval is not None:
        lookup = {s.section_id: s for s in state.retrieval.sections if s.section_id}

    names: list[str] = []
    for section_id in verified:
        section = lookup.get(section_id)
        if section is not None and section.section_number and section.act_name:
            names.append(f"Section {section.section_number} of the {section.act_name}")
        else:
            names.append(section_id)
    return names


def _citations_line(state: LegalQueryState, verified: list[str], persona: str) -> str:
    if persona == LAWYER:
        if verified:
            return "Verified citations: " + ", ".join(verified)
        return "Verified citations: none"
    # Citizen: friendlier label, and readable section names instead of the
    # machine ids, which a layperson cannot look up or use.
    if verified:
        return "The law behind this answer: " + "; ".join(
            _display_names(state, verified)
        )
    return "No specific sections could be confirmed for your question."


def _format_answer(state: LegalQueryState, partial: bool, persona: str) -> str:
    verified = state.verified_section_ids or []
    body = _render_markers(
        state.draft_answer or "",
        set(verified),
        drop_verified=(persona == CITIZEN),
    )
    if not body:
        body = (
            "No answer text was produced for this query, and no sections could "
            "be cited."
        )

    parts: list[str] = []
    if partial:
        gap = ""
        if state.sufficiency is not None and state.sufficiency.missing:
            gap = f" The retrieved law does not fully cover: {state.sufficiency.missing}."
        if persona == LAWYER:
            parts.append(
                "PARTIAL ANSWER — the retrieved sections only partially address "
                "this query." + gap
            )
        else:
            parts.append(
                "A heads-up: the law found here only partially answers your "
                "question." + gap
            )
        parts.append("")

    parts.append(body)
    parts.append("")
    parts.append(_citations_line(state, verified, persona))
    parts.append(_confidence_line(state, persona, partial))
    if state.disclaimer:
        parts.append("")
        parts.append(state.disclaimer)
    return "\n".join(parts)


def _format_out_of_domain(state: LegalQueryState) -> str:
    return (
        "This question appears to fall outside the scope of this system.\n\n"
        f"{_SCOPE_LINE}\n\n"
        "No legal sections were retrieved or cited for this query, so the system "
        "cannot answer it."
    )


def _format_rejected(state: LegalQueryState) -> str:
    return (
        "This request cannot be answered.\n\n"
        "It appears to seek help in avoiding or defeating a legal obligation owed "
        "to an employee, which is outside what this system will assist with.\n\n"
        "If you are an employee trying to understand your rights, please rephrase "
        "the question from that perspective and the system can help."
    )


def _format_error(state: LegalQueryState) -> str:
    detail = state.error or "an unspecified internal error"
    return (
        "The system encountered an internal error while processing this query and "
        "could not produce a complete answer.\n\n"
        f"Details: {detail}\n\n"
        "Please try again. No legal conclusions should be drawn from this response."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def final_response_node(state: LegalQueryState) -> dict:
    try:
        status = _resolve_status(state)
        persona = normalize_persona(state.persona)

        if status in _NO_CONTENT_STATUSES:
            if status == "error":
                final_answer = _format_error(state)
            elif status == "rejected":
                final_answer = _format_rejected(state)
            else:
                final_answer = _format_out_of_domain(state)

            # The fixed message above already carries no legal content, but the
            # CITATIONS THEMSELVES must be dropped from the state too, not just
            # from the rendered text. api/routes/query.py returns
            # verified_section_ids straight to the caller, so leaving them set
            # would hand an API consumer the section list for a query we just
            # refused to answer — the guardrail's work undone at the boundary.
            # The confidence score goes with them: it scores an answer that is
            # not being given.
            return {
                "status": status,
                "final_answer": final_answer,
                "verified_section_ids": [],
                "confidence": 0.0,
                "confidence_factors": {},
            }

        if status == "insufficient_evidence":
            final_answer = _format_answer(state, partial=True, persona=persona)
        else:  # "ok"
            final_answer = _format_answer(state, partial=False, persona=persona)

        return {"status": status, "final_answer": final_answer}

    except Exception as exc:  # noqa: BLE001 — this node must never raise.
        logger.exception("final_response_node failed; returning safe fallback")
        return {
            "status": "error",
            "final_answer": _SAFE_FALLBACK,
            "error": state.error or f"final_response_node failed: {exc}",
        }
