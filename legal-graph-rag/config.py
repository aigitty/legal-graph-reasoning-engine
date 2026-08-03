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

    # Final sections cap for the evidence pack, applied AFTER the union across
    # all grounded concepts has been ranked against the user's query.
    MAX_SECTIONS: int = 15

    # Per-concept safety valve inside graph/traversal.py. Deliberately much
    # larger than MAX_SECTIONS: trimming per concept happens before the ranker
    # has seen the query, so cutting hard there would discard candidates the
    # ranker would have chosen. Global ranking then cuts to MAX_SECTIONS.
    MAX_SECTIONS_PER_CONCEPT: int = 40

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
    # Retrieval ranking (graph/ranking.py)                                 #
    #                                                                      #
    # Deterministic relevance scoring over the retrieved set. Before this, #
    # sections came back in (act_id, section_number) order and the cap     #
    # sliced that arbitrary order, so the sections that actually answered  #
    # the question could be dropped. Weights are normalised by their sum,  #
    # so they do not have to add to 1.0.                                   #
    # ------------------------------------------------------------------ #
    RANK_W_RELEVANCE: float = 0.35      # ontology said primary vs supporting
    RANK_W_CONCEPT_HITS: float = 0.15   # how many grounded concepts hit it
    RANK_W_HOP: float = 0.20            # anchor beats a distant CITES hop
    RANK_W_LEXICAL: float = 0.20        # BM25 of the query vs title + text
    RANK_W_ACT_PRIORITY: float = 0.10   # prefer the operative consolidating Code

    # Score floor for a SUPPORTING section on the relevance signal. Not 0.0:
    # a supporting section is still curated evidence, it just should not
    # outrank a primary on that signal alone.
    RANK_SUPPORTING_FLOOR: float = 0.45

    # How many times the section title is repeated into the BM25 document.
    # A title match is a much stronger signal of topicality than one hit
    # somewhere in a 19,000-character section body.
    RANK_TITLE_WEIGHT: int = 3

    # ------------------------------------------------------------------ #
    # Temporal + territorial filtering (A.2 / A.3)                         #
    #                                                                      #
    # Source of truth is data/ontology/act_metadata.json, mirrored onto    #
    # the Act nodes by ingest/act_metadata_loader.py.                      #
    # ------------------------------------------------------------------ #

    # Drop sections belonging to a repealed Act before they reach the LLM.
    # Turning this off restores the pre-fix behaviour (repealed and operative
    # law cited side by side) and is only useful for A/B demonstration.
    SUPPRESS_REPEALED_ACTS: bool = True

    # Drop a STATE act when the user named a DIFFERENT state. Karnataka's
    # Shops Act must never be cited to a user in Maharashtra.
    FILTER_BY_JURISDICTION: bool = True

    # When the user names no state at all, state acts are kept (dropping them
    # would leave weekly-holiday / annual-leave questions unanswerable) but
    # multiplied by this factor so Central law ranks first, and a warning is
    # raised. 0.0 would drop them entirely.
    UNSTATED_JURISDICTION_PENALTY: float = 0.6

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

    # ------------------------------------------------------------------ #
    # Persona-aware output                                                 #
    #                                                                      #
    # Selected at login; tailors ONLY how the answer is written, never    #
    # what is retrieved or cited. Canonical values live in                #
    # agents/persona.py ("citizen" | "lawyer"). "citizen" is the safer    #
    # default (see agents/persona.normalize_persona).                     #
    # ------------------------------------------------------------------ #
    DEFAULT_PERSONA: str = "citizen"

    # --- Citizen readability controls (presentation only) ---------------- #
    # Optional cap on how many sections the synthesis LLM is shown for the
    # citizen persona (primaries first). DEFAULT 0 = disabled, deliberately.
    #
    # Trimming was tried as a way to stop 15-section packs producing
    # section-by-section citizen answers, but supporting sections arrive in
    # BFS/act order rather than relevance order, so a cap of 5 silently dropped
    # the sections that actually answered the question (a deductions query lost
    # COW_2019_S18/S21 and had to answer "the law does not address this"). The
    # prompt's own "use at most 3 sections" rule achieves the same brevity
    # without discarding evidence. Keep this at 0 unless a relevance-ranked
    # ordering exists to trim against.
    CITIZEN_EVIDENCE_PACK_LIMIT: int = 0

    # Bands for the citizen trailer's worded confidence ("high"/"moderate"/
    # "low"). The lower band mirrors MIN_CONFIDENCE.
    CITIZEN_CONFIDENCE_HIGH: float = 0.7
    CITIZEN_CONFIDENCE_MODERATE: float = 0.4


# Module-level singleton — import this everywhere.
cfg = Settings()
