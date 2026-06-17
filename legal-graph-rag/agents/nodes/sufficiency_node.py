"""
agents/nodes/sufficiency_node.py

Second Gemini call: sufficiency evaluation (architecture section 7).

Given the query, extracted entities, and a lightweight summary of retrieved
sections (titles + short previews, not full text), returns whether the
evidence is sufficient and, if not, what's missing.

LOOP TERMINATION IS NEVER DECIDED BY THE LLM: if retrieval_iterations has
already reached MAX_ITERATIONS, this node returns "exhausted" deterministically
without calling Gemini. No expansion loop exists yet — this guard is in place
ahead of that step.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.graph_state import LegalQueryState
from agents.llm import get_llm
from agents.schemas import SufficiencyVerdict
from agents.state import SectionContext
from config import cfg

logger = logging.getLogger(__name__)

MAX_ITERATIONS = cfg.MAX_RETRIEVAL_ITERATIONS
PREVIEW_CHARS = cfg.SUFFICIENCY_PREVIEW_CHARS

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sufficiency.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

_SUFFICIENCY_LLM = get_llm(
    temperature=cfg.SUFFICIENCY_TEMPERATURE,
    max_output_tokens=cfg.SUFFICIENCY_MAX_TOKENS,
).with_structured_output(SufficiencyVerdict)


def _summarize_section(section: SectionContext) -> str:
    preview = " ".join(section.section_text.split())[:PREVIEW_CHARS]
    return (
        f"- [{section.relevance.upper()}] {section.section_id} "
        f"({section.act_name}) — {section.section_title}: {preview}..."
    )


def sufficiency_evaluator_node(state: LegalQueryState) -> dict:
    # Deterministic loop-termination guard — checked BEFORE any LLM call.
    if state.retrieval_iterations >= MAX_ITERATIONS:
        return {
            "sufficiency": SufficiencyVerdict(
                sufficient=False, missing="iteration_limit_reached"
            ),
            "warnings": state.warnings + ["Max retrieval iterations reached (exhausted)."],
        }

    if state.retrieval is None or state.retrieval.is_empty:
        return {
            "sufficiency": SufficiencyVerdict(
                sufficient=False,
                missing="No sections were retrieved from the graph for this query.",
            ),
            "retrieval_iterations": state.retrieval_iterations + 1,
        }

    sections_summary = "\n".join(
        _summarize_section(s) for s in state.retrieval.sections
    )

    extraction_summary = "none"
    if state.extraction:
        e = state.extraction
        extraction_summary = (
            f"jurisdiction={e.jurisdiction}, employment_type={e.employment_type}, "
            f"years_of_service={e.years_of_service}, triggering_event={e.triggering_event}"
        )

    human_message = (
        f"User query: {state.raw_query}\n"
        f"Extracted entities: {extraction_summary}\n\n"
        f"Retrieved sections:\n{sections_summary}"
    )

    try:
        verdict: SufficiencyVerdict = _SUFFICIENCY_LLM.invoke(
            [("system", SYSTEM_PROMPT), ("human", human_message)]
        )
    except Exception as exc:
        logger.exception("Sufficiency evaluation failed")
        # Degrade gracefully: don't block the pipeline on this call failing.
        return {
            "sufficiency": SufficiencyVerdict(sufficient=True, missing=""),
            "warnings": state.warnings + [f"Sufficiency evaluation failed, defaulting to sufficient: {exc}"],
            "retrieval_iterations": state.retrieval_iterations + 1,
        }

    return {"sufficiency": verdict, "retrieval_iterations": state.retrieval_iterations + 1}