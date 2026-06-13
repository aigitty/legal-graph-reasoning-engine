"""
agents/graph_state.py

LangGraph channel state for the Legal Graph RAG agent workflow.

This is deliberately separate from agents/state.py:
- agents/state.py      -> dependency-free DOMAIN dataclasses (RetrievalResult, etc.)
- agents/graph_state.py -> the Pydantic ORCHESTRATION state that LangGraph
  passes from node to node.

Step 2 adds:
  - grounded_concepts: canonical Concept.name values from concept grounding
  - warnings: non-fatal notices (e.g. "no concept matched")
  
Step 3 adds:
  - extraction: structured Gemini output (ExtractionResult)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.state import RetrievalResult
from agents.schemas import ExtractionResult, SufficiencyVerdict



class LegalQueryState(BaseModel):
    # arbitrary_types_allowed lets this Pydantic model carry the existing
    # stdlib dataclass (RetrievalResult) without converting/validating it.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_query: str = ""
    concept: str = ""  # legacy direct-concept override; superseded by grounding
    extraction: Optional[ExtractionResult] = None
    grounded_concepts: list[str] = Field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    sufficiency: Optional[SufficiencyVerdict] = None
    retrieval_iterations: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None