"""
agents/entity_extractor/agent.py

Entity extractor agent — Agent 1 in the legal reasoning pipeline.

Responsibility:
    Takes a raw user query and returns a structured ExtractedEntities
    dataclass containing legal concepts, jurisdiction, entity type,
    and confidence score.

Uses:
    Google ADK LlmAgent with Gemini Flash model.
    helper.py handles all prompt building and response parsing.
    No domain logic lives here — only agent wiring.
"""

import asyncio
import logging
import uuid
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google import genai
from google.genai import types

from agents.entity_extractor.helper import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    parse_gemini_response,
    to_extracted_entities,
)
from agents.state import ExtractedEntities

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────

entity_extractor_agent = LlmAgent(
    name="entity_extractor",
    model="gemini-2.5-flash",
    instruction=ENTITY_EXTRACTION_SYSTEM_PROMPT,
    description="Extracts legal entities and concepts from a plain-language query.",
)
root_agent = entity_extractor_agent

# ─────────────────────────────────────────────
# Runner setup
# ─────────────────────────────────────────────

APP_NAME = "legal_graph_rag"

session_service = InMemorySessionService()

runner = Runner(
    agent=entity_extractor_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# ─────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────

async def extract_entities(raw_query: str) -> ExtractedEntities:
    """
    Run the entity extractor agent on a raw user query.

    Creates a fresh session for each call so there is no
    conversation history bleeding between requests.

    Parameters:
        raw_query: Plain-language legal query from the user.

    Returns:
        ExtractedEntities dataclass with concepts, jurisdiction,
        entity type, confidence, and validity flag.
    """
    session_id = str(uuid.uuid4())
    user_id = "legal_rag_user"

    # Create a fresh session for this query
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    # Build the user-turn message
    
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=build_extraction_prompt(raw_query))],
    )

    # Run the agent and collect the final response text
    response_text = ""
    async for event in runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=user_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text

    logger.debug("Raw Gemini response: %s", response_text[:200])

    # Parse and map to dataclass
    parsed = parse_gemini_response(response_text)
    entities = to_extracted_entities(parsed, raw_query)

    logger.info(
        "Extraction complete — concepts=%s jurisdiction=%s valid=%s confidence=%s",
        entities.concepts,
        entities.jurisdiction,
        entities.is_valid,
        entities.confidence,
    )

    return entities


# ─────────────────────────────────────────────
# Entry point — test independently
# ─────────────────────────────────────────────

async def _test():
    test_queries = [
        "I was fired without notice after 3 years at a private company in Karnataka",
        "my employer has not paid salary for 2 months",
        "am I eligible for gratuity after 5 years",
        "what is the weather today",   # out of domain
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await extract_entities(query)
        print(f"  concepts:     {result.concepts}")
        print(f"  jurisdiction: {result.jurisdiction}")
        print(f"  entity_type:  {result.entity_type}")
        print(f"  years:        {result.years_of_service}")
        print(f"  confidence:   {result.confidence}")
        print(f"  is_valid:     {result.is_valid}")
        if not result.is_valid:
            print(f"  note:         {result.validation_note}")


if __name__ == "__main__":
    asyncio.run(_test())