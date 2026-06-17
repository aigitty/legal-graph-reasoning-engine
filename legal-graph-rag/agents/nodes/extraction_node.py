"""
agents/nodes/extraction_node.py

First Gemini call in the pipeline: structured entity/concept/jurisdiction
extraction (architecture section 9). Output feeds concept_grounding_node.

in_domain / safety_flag are captured but NOT acted on yet — conditional
routing on them is part of the guardrails step (not this one).
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.graph_state import LegalQueryState
from agents.llm import get_llm
from agents.schemas import ExtractionResult
from config import cfg

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extraction.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# Built once at import time. Requires .env already loaded (graph_agent.py
# calls load_dotenv() before importing nodes).
_EXTRACTION_LLM = get_llm(
    temperature=cfg.EXTRACTION_TEMPERATURE,
    max_output_tokens=cfg.EXTRACTION_MAX_TOKENS,
).with_structured_output(ExtractionResult)


def entity_extraction_node(state: LegalQueryState) -> dict:
    if not state.raw_query.strip():
        return {"error": "Empty raw_query."}

    try:
        result: ExtractionResult = _EXTRACTION_LLM.invoke(
            [("system", SYSTEM_PROMPT), ("human", state.raw_query)]
        )
    except Exception as exc:
        logger.exception("Entity extraction failed")
        return {"error": f"Entity extraction failed: {exc}"}

    return {"extraction": result}