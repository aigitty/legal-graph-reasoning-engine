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

from agents.calculators import Calculation, format_inr
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
        # is_unsafe, not raw truthiness: safety_flag holds a free-text REASON,
        # so the sentinel strings a model returns for "nothing to report"
        # ("none", "null") would otherwise refuse a perfectly legitimate query
        # and show the user no legal content at all.
        if extraction.is_unsafe:
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
    #
    # The word is followed by what it actually MEANS. "How confident is this
    # answer: high" was read as "this answer is legally correct for my
    # situation", which is not what the score measures — it measures how well
    # the law was grounded, retrieved and citation-checked. On a page about
    # losing your job that gap matters, so the qualifier is not optional.
    word = _confidence_word(state.confidence, _evidence_is_partial(state, partial))
    return (
        f"How reliable is this: {word}.\n"
        "That describes how well the law was found and checked — it is not a "
        "prediction of what will happen in your case."
    )


def _act_ids_for(verified: list[str]) -> list[str]:
    """Distinct act_ids behind the verified citations, in first-cited order."""
    act_ids: list[str] = []
    for section_id in verified:
        act_id = act_registry.act_id_for_section(section_id)
        if act_id and act_id not in act_ids:
            act_ids.append(act_id)
    return act_ids


def _commencement_line(verified: list[str], persona: str) -> str:
    """
    State WHEN the cited law came into force.

    An answer that gives the current position without dating it is unsafe on the
    queries where dating matters most: the three Labour Codes commenced on
    21 November 2025, so conduct before that date is still governed by the Acts
    this engine now suppresses. Neither persona was told this, which meant a
    practitioner could apply the IRC to a pre-commencement dispute and a worker
    could not tell whether the answer covered what happened to them.

    Deterministic — read from act_metadata.json via act_registry, never from the
    model.
    """
    notes = act_registry.commencement_notes(_act_ids_for(verified))
    if not notes:
        return ""

    listed = "; ".join(f"{name} — {when}" for name, when in notes)
    if persona == LAWYER:
        return (
            f"In force from: {listed}. Conduct predating commencement may remain "
            "governed by the corresponding repealed enactment."
        )
    if len(notes) == 1:
        name, when = notes[0]
        body = f"The {name} came into force on {when}."
    else:
        body = "The law cited above came into force on these dates — " + listed + "."
    return (
        f"When this law started applying: {body} If the events in question "
        "happened before that date, the earlier law may apply instead, so "
        "mention the dates when you get help."
    )


def _superseded_line(
    state: LegalQueryState, persona: str, verified: list[str]
) -> str:
    """
    Tell the user which law was WITHHELD as repealed, and what replaced it.

    The suppression was already recorded in retrieval.suppressed_acts and pushed
    into state.warnings, but warnings are never displayed — so the user never
    learned that the Act every search result still describes has been replaced.
    A correct answer that contradicts everything published before November 2025
    reads as a wrong answer unless it says why.

    RELEVANCE FILTER: only Acts whose SUCCESSOR was actually cited are
    mentioned. suppressed_acts accumulates across every concept traversed,
    including the remedy layer, so a wages question was reporting that the
    Industrial Disputes Act and the Payment of Gratuity Act had been repealed —
    true, unprompted, and nothing to do with what the reader asked. A note about
    superseded law only helps when it explains the law that is actually in the
    answer.
    """
    if state.retrieval is None or not state.retrieval.suppressed_acts:
        return ""

    cited_acts = set(_act_ids_for(verified))
    if not cited_acts:
        return ""

    relevant = [
        act_id
        for act_id in state.retrieval.suppressed_acts
        if (meta := act_registry.get_act(act_id)) is not None
        and meta.repealed_by in cited_acts
    ]
    if not relevant:
        return ""

    if persona == LAWYER:
        notes = [n for n in (act_registry.repeal_note(a) for a in relevant) if n]
        return "Excluded as repealed: " + " ".join(notes) if notes else ""

    notes = [n for n in (act_registry.replacement_note(a) for a in relevant) if n]
    if not notes:
        return ""
    return (
        "If you search this online: " + " ".join(notes) + " Older articles and "
        "websites may still describe the earlier rules."
    )


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


def _temporal_mismatch_banner(state: LegalQueryState, persona: str) -> str:
    """
    A deterministic, un-skippable warning when the query's own stated event
    date predates the commencement of the Act the answer relies on.

    THIS IS THE SAFETY NET, not the only mechanism. synthesis_node already asks
    the model to address this in its opening — but this project's entire
    integrity model (CLAUDE.md) is that a prompt ASKS and deterministic code
    ENFORCES, because a model can be asked to do something correctly and simply
    not do it. Citation verification does not trust the model to only cite what
    it was shown; this does not trust the model to remember to mention a
    mismatch it was told about three paragraphs of instructions earlier. It is
    prepended BEFORE the model's own opening line, because a warning placed
    after a confident paragraph is a warning most readers have already stopped
    reading for.
    """
    conflicts = state.temporal_conflicts
    if not conflicts:
        return ""

    if persona == LAWYER:
        lines = []
        for act_name, when, predecessor in conflicts:
            older = f" {predecessor} may govern instead." if predecessor else ""
            lines.append(f"{act_name} (in force from {when}).{older}")
        return (
            "TEMPORAL MISMATCH: the stated event date precedes the "
            "commencement of " + " ".join(lines) + " The analysis below cites "
            "currently in-force sections, but their applicability to facts "
            "predating commencement has NOT been verified and should not be "
            "assumed."
        )

    names = [act_name for act_name, _, _ in conflicts]
    listed = names[0] if len(names) == 1 else ", ".join(names)
    return (
        f"Before anything else: you mentioned this happened before the "
        f"{listed} existed. The law changed on 21 November 2025, so an older "
        f"law may actually cover your situation instead of what is explained "
        f"below. Please double-check the date, or mention it when you get "
        f"advice — the numbers below may not be the ones that applied to you."
    )


def _display_name(state: LegalQueryState, section_id: str) -> str:
    """"Section 53 of the Code on Social Security, 2020" for a calc's citation."""
    if state.retrieval is not None:
        for section in state.retrieval.sections:
            if section.section_id == section_id and section.section_number and section.act_name:
                return f"Section {section.section_number} of the {section.act_name}"
    return section_id


def _render_calculation(state: LegalQueryState, calc: Calculation, persona: str) -> str:
    """
    One Calculation as a clearly-separated block, deterministically formatted —
    never left to the model to restate, so the number the user sees is always
    exactly the one Python computed (see synthesis_node's calc_note).
    """
    where = _display_name(state, calc.section_id)

    if not calc.is_computed:
        # The honest-gap case: still shown, not hidden, because "we found the
        # rule but cannot safely compute a number from it" is itself useful
        # information (agents/calculators.retrenchment_compensation_gap).
        return f"{calc.label} — not calculated ({where}): {calc.note}"

    amount = format_inr(calc.amount)
    if persona == LAWYER:
        lines = [f"{calc.label}: {amount}  [{where}: {calc.formula}]"]
        lines += [f"  Note: {a}" for a in calc.assumptions]
        return "\n".join(lines)

    lines = [f"{calc.label}: approximately {amount}"]
    lines.append(f"  Based on: {where}")
    lines += [f"  Note: {a}" for a in calc.assumptions]
    return "\n".join(lines)


def _calculations_block(state: LegalQueryState, persona: str) -> str:
    calcs = state.verified_calculations or []
    if not calcs:
        return ""
    heading = "Estimated amounts" if persona != LAWYER else "Calculated entitlements"
    body = "\n\n".join(_render_calculation(state, c, persona) for c in calcs)
    footer = (
        "These are estimates from the figures you gave and the exact "
        "statutory formula — not a substitute for your employer's or a "
        "labour authority's own calculation."
    )
    return f"{heading}:\n{body}\n\n{footer}"


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
    banner = _temporal_mismatch_banner(state, persona)
    if banner:
        parts.append(banner)
        parts.append("")
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

    calc_block = _calculations_block(state, persona)
    if calc_block:
        parts.append("")
        parts.append(calc_block)

    parts.append("")
    parts.append(_citations_line(state, verified, persona))

    # Temporal context, both deterministic. Placed with the citations rather
    # than in the body because they qualify WHICH law was cited, and because
    # leaving them to the LLM would make them optional.
    commencement = _commencement_line(verified, persona)
    if commencement:
        parts.append(commencement)

    superseded = _superseded_line(state, persona, verified)
    if superseded:
        parts.append(superseded)

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
                "verified_calculations": [],
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
