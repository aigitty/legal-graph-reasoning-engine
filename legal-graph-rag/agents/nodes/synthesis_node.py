"""
agents/nodes/synthesis_node.py

THIRD and final Gemini call: answer synthesis (architecture section 6, node 7).

Given the user's query, the extracted entities, and the FULL text of the
sections returned by the deterministic graph traversal, produce a
human-readable "reasoned legal path" answer that cites sections using a
parseable [SECTION_ID] marker.

CORE GUARANTEE THIS NODE MUST PRESERVE:
The LLM may only describe and cite sections already present in the evidence
pack (state.retrieval.sections). It never invents, selects, or recalls a
section from its own knowledge. This node ASKS for that rule via the prompt;
it is ENFORCED deterministically in the later output-guardrail step. So this
node only:
  - produces draft_answer (free text with [SECTION_ID] markers), and
  - parses those markers into cited_section_ids (no filtering, no DB check —
    that is the output guardrail's job).

Determinism boundaries:
  - If the graph returned NO sections (out-of-domain / empty retrieval), this
    node short-circuits with a fixed honest "no applicable sections found"
    message and ZERO citations, WITHOUT calling Gemini. This keeps the empty
    path provably hallucination-free.
  - Otherwise it makes exactly one Gemini call (with a single retry on error).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agents.graph_state import LegalQueryState
from agents.llm import get_llm
from agents.state import SectionContext
from config import cfg

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "synthesis.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# Free text (no structured output). Slightly warmer than the structured calls
# (architecture section 9: synthesis temp 0.2), with a larger token budget
# because this is the only long-form output in the pipeline.
#
# NOTE: gemini-2.5-flash has "thinking" enabled by default on Vertex AI, and
# thinking tokens are drawn from the SAME max_output_tokens budget as the
# visible answer. With dynamic thinking, larger evidence packs caused MORE
# of the budget to go to thinking, truncating the visible answer earlier
# (observed: a 12-section pack truncated after one clause even at 8192,
# while a 6-section pack got most of the way through). thinking_budget=0
# disables this for synthesis, which is a constrained
# transformation/formatting task that doesn't need extended reasoning —
# the full budget then goes to the visible, citation-bearing answer.
_SYNTHESIS_LLM = get_llm(
    temperature=cfg.SYNTHESIS_TEMPERATURE,
    max_output_tokens=cfg.SYNTHESIS_MAX_TOKENS,
    thinking_budget=cfg.SYNTHESIS_THINKING_BUDGET,
)

# Honest fallback for the empty-retrieval / out-of-domain path. No markers ever.
_NO_EVIDENCE_ANSWER = (
    "The system did not find any applicable sections in its legal knowledge "
    "graph for this query.\n\n"
    "This engine answers questions about Indian employment and labour law as "
    "covered by the Industrial Disputes Act 1947, the Payment of Gratuity Act "
    "1972, the Minimum Wages Act 1948, the Karnataka Shops and Establishments "
    "Act 1961, and the Code on Wages 2019. If your question falls outside that "
    "scope, the system cannot answer it.\n\n"
    "The system does not guess at legal provisions it has not retrieved, so no "
    "sections are cited here."
)

# Permissive citation-marker pattern. Deliberately NOT restricted to the valid
# section_id shape: catching malformed / out-of-pack markers (e.g. [IPC/302])
# is the entire point of verification and the later output guardrail. Matches an
# uppercase/digit-led token (no internal spaces) inside square brackets.
_CITATION_RE = re.compile(r"\[([A-Z0-9][A-Z0-9_/.\-]{2,})\]")


def _looks_like_citation(token: str) -> bool:
    # Exclude bare label words like [PRIMARY]; real section ids carry a digit,
    # an underscore, or a slash.
    return any(ch.isdigit() for ch in token) or "_" in token or "/" in token


def _parse_citations(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in _CITATION_RE.findall(text):
        if _looks_like_citation(token) and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _format_section(section: SectionContext) -> str:
    full_text = " ".join(section.section_text.split())
    return (
        f"[{section.section_id}] ({section.relevance.upper()}) "
        f"{section.act_name} — Section {section.section_number}: "
        f"{section.section_title}\n{full_text}"
    )


def _build_human_message(state: LegalQueryState) -> str:
    sections = state.retrieval.sections
    pack = "\n\n".join(_format_section(s) for s in sections)

    grounded = ", ".join(state.grounded_concepts) or "none"

    if state.extraction:
        e = state.extraction
        details = (
            f"- grounded legal concepts: {grounded}\n"
            f"- jurisdiction: {e.jurisdiction}\n"
            f"- employment type: {e.employment_type}\n"
            f"- years of service: {e.years_of_service}\n"
            f"- triggering event: {e.triggering_event}"
        )
    else:
        details = f"- grounded legal concepts: {grounded}"

    if state.sufficiency:
        suff = (
            f"sufficient={state.sufficiency.sufficient}; "
            f"missing={state.sufficiency.missing!r}"
        )
    else:
        suff = "unknown"

    return (
        f"User query:\n{state.raw_query}\n\n"
        f"Extracted details:\n{details}\n\n"
        f"Sufficiency verdict: {suff}\n\n"
        f"EVIDENCE PACK ({len(sections)} sections):\n{pack}"
    )


def answer_synthesis_node(state: LegalQueryState) -> dict:
    # Deterministic empty-evidence path: no Gemini call, no citations.
    if state.retrieval is None or not state.retrieval.sections:
        return {"draft_answer": _NO_EVIDENCE_ANSWER, "cited_section_ids": []}

    messages = [("system", SYSTEM_PROMPT), ("human", _build_human_message(state))]

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            response = _SYNTHESIS_LLM.invoke(messages)
            draft = getattr(response, "content", str(response)).strip()
            return {
                "draft_answer": draft,
                "cited_section_ids": _parse_citations(draft),
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Answer synthesis attempt %s failed: %s", attempt, exc)

    logger.error("Answer synthesis failed after retry: %s", last_exc)
    return {
        "draft_answer": (
            "The system was unable to generate an answer due to an internal "
            "error while contacting the reasoning model. Please try again."
        ),
        "cited_section_ids": [],
        "error": f"Answer synthesis failed: {last_exc}",
    }