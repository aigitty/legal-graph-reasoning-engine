"""
agents/nodes/extraction_node.py

First Gemini call in the pipeline: structured entity/concept/jurisdiction
extraction (architecture section 9). Output feeds concept_grounding_node.

in_domain / safety_flag are captured but NOT acted on yet — conditional
routing on them is part of the guardrails step (not this one).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from agents.graph_state import LegalQueryState
from agents.llm import get_llm
from agents.schemas import ExtractionResult
from config import cfg
from graph import act_registry

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extraction.txt"

# The corpus list is INJECTED from act_metadata.json rather than written into
# the prompt, so it can never drift from the Acts the engine actually holds.
# It had already drifted: the file still named the Minimum Wages Act 1948 and
# the Payment of Gratuity Act 1972 as live and did not mention the Industrial
# Relations Code 2020 or the Code on Social Security 2020 at all — so the first
# LLM call was extracting against a corpus three Acts out of date. The prompt
# itself remains a file (CLAUDE.md rule 7); only this factual block is composed.
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").replace(
    "{ACTS_IN_FORCE}", act_registry.corpus_block()
)

# Built once at import time. Requires .env already loaded (graph_agent.py
# calls load_dotenv() before importing nodes).
_EXTRACTION_LLM = get_llm(
    temperature=cfg.EXTRACTION_TEMPERATURE,
    max_output_tokens=cfg.EXTRACTION_MAX_TOKENS,
).with_structured_output(ExtractionResult)


def entity_extraction_node(state: LegalQueryState) -> dict:
    if not state.raw_query.strip():
        return {"error": "Empty raw_query."}

    # Today's date is given alongside the query, never baked into the system
    # prompt, so relative phrases ("last August", "two years ago") resolve to
    # the right YEAR rather than whatever year the prompt happened to be
    # written in. Without this, event_date extraction for relative phrases is a
    # coin flip on the correct year — and the year is the one thing that
    # decides whether the pre- or post-21-November-2025 law applies.
    human_message = (
        f"Today's date: {date.today().isoformat()}\n\nQuery:\n{state.raw_query}"
    )

    try:
        result: ExtractionResult = _EXTRACTION_LLM.invoke(
            [("system", SYSTEM_PROMPT), ("human", human_message)]
        )
    except Exception as exc:
        logger.exception("Entity extraction failed")
        return {"error": f"Entity extraction failed: {exc}"}

    return {"extraction": result}