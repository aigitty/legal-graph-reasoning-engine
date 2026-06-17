"""
config.py

Single source of truth for all runtime configuration in the Legal Graph RAG
pipeline. Every hardcoded constant that could reasonably vary across
environments (model names, temperatures, limits, credentials) lives here.

Usage
-----
    from config import cfg

    cfg.GEMINI_MODEL          # "gemini-2.5-flash"
    cfg.NEO4J_URI             # from .env
    cfg.MAX_RETRIEVAL_ITERATIONS  # 2

Settings are loaded in this priority order (highest wins):
    1. Actual environment variables (set in the shell or CI secrets)
    2. Values in the .env file at the project root
    3. Defaults defined in this file

Adding a new setting
--------------------
Add a field with a type annotation and a default. It is automatically
overridable via an env var of the SAME name. No other changes needed.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",      # silently ignore unrecognised .env keys
        case_sensitive=False,  # NEO4J_URI == neo4j_uri
    )

    # ------------------------------------------------------------------ #
    # Neo4j                                                                #
    # ------------------------------------------------------------------ #
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    # ------------------------------------------------------------------ #
    # Google Cloud / Vertex AI                                             #
    # ------------------------------------------------------------------ #
    # Auth is via ADC (gcloud auth application-default login). Do NOT set
    # GOOGLE_APPLICATION_CREDENTIALS. See agents/llm.py for the full note.
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"

    # ------------------------------------------------------------------ #
    # LangSmith observability (optional — tracing only activates when     #
    # LANGCHAIN_TRACING_V2=true is set and LANGCHAIN_API_KEY is present)  #
    # ------------------------------------------------------------------ #
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "legal-graph-rag"

    # ------------------------------------------------------------------ #
    # LLM — model                                                          #
    # ------------------------------------------------------------------ #
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ------------------------------------------------------------------ #
    # LLM — per-call settings                                              #
    #                                                                      #
    # Extraction & sufficiency use structured output (JSON) so they need  #
    # temperature=0 and a modest token budget. Synthesis is the only      #
    # long-form free-text call; it gets a larger budget and               #
    # thinking_budget=0 to prevent Gemini's "thinking" tokens from eating #
    # into the visible answer (observed truncation at 8192 without this). #
    # ------------------------------------------------------------------ #
    EXTRACTION_TEMPERATURE: float = 0.0
    EXTRACTION_MAX_TOKENS: int = 1024

    SUFFICIENCY_TEMPERATURE: float = 0.0
    SUFFICIENCY_MAX_TOKENS: int = 1024

    SYNTHESIS_TEMPERATURE: float = 0.2
    SYNTHESIS_MAX_TOKENS: int = 8192
    SYNTHESIS_THINKING_BUDGET: int = 0   # 0 = disable extended thinking

    # ------------------------------------------------------------------ #
    # Pipeline hard limits                                                 #
    #                                                                      #
    # These are enforced deterministically in Python BEFORE any LLM call  #
    # (CLAUDE.md rule 4 — loop termination is never decided by the LLM).  #
    # ------------------------------------------------------------------ #

    # Max extra retrieval passes after the first traversal.
    # sufficiency_node exits the loop when retrieval_iterations >= this.
    MAX_RETRIEVAL_ITERATIONS: int = 2

    # Sections cap per traversal call (retrieval_node).
    MAX_SECTIONS: int = 15

    # Maximum traversal depth. Initial depth is MAX_HOPS_DEFAULT; each
    # expansion pass adds one hop up to MAX_HOPS_CAP.
    MAX_HOPS_DEFAULT: int = 2
    MAX_HOPS_CAP: int = 4

    # LangGraph backstop — prevents infinite recursion if a bug bypasses
    # the deterministic loop guard.
    LANGGRAPH_RECURSION_LIMIT: int = 25

    # Sufficiency-preview character limit used in the sufficiency prompt.
    SUFFICIENCY_PREVIEW_CHARS: int = 200

    # ------------------------------------------------------------------ #
    # Output guardrail — confidence                                        #
    #                                                                      #
    # Weights must sum to 1.0. Formula (CLAUDE.md section 6):             #
    #   confidence = W_CONCEPT  * concept_coverage                        #
    #              + W_SEED     * seed_strength                           #
    #              + W_SUFFICIENCY * sufficiency_score                    #
    #              + W_CITATION * citation_validity                       #
    # ------------------------------------------------------------------ #
    CONFIDENCE_W_CONCEPT: float = 0.35
    CONFIDENCE_W_SEED: float = 0.25
    CONFIDENCE_W_SUFFICIENCY: float = 0.20
    CONFIDENCE_W_CITATION: float = 0.20

    # Below this threshold the answer status is downgraded to
    # "insufficient_evidence".
    MIN_CONFIDENCE: float = 0.4


# Module-level singleton — import this everywhere.
cfg = Settings()
