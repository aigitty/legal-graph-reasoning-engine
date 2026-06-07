import json
import logging
import re

from agents.state import ExtractedEntities


logger = logging.getLogger(__name__)


ENTITY_EXTRACTION_SYSTEM_PROMPT: str = """
You are a legal entity extractor for Indian employment and labour law.

Your job is to read the user's query and extract structured legal information from it.

Extract the following fields and return them as a JSON object:

- concepts: list of legal concepts present in the query.
  Examples: "wrongful termination", "notice period", "gratuity", "minimum wages",
  "retrenchment", "salary delay", "overtime", "lay-off", "bonus", "strike",
  "working hours", "annual leave", "factory closure", "unfair labour practice"
  Return an empty list if no legal concept is identifiable.

- jurisdiction: the Indian state or "Central" if no state is mentioned.
  Examples: "Karnataka", "Maharashtra", "Tamil Nadu", "Central"
  Default to "Central" if not mentioned.

- entity_type: who the user is. One of: "employee", "employer", "consumer", "union"
  Default to "employee" if not clear.

- years_of_service: number of years the user has worked, as a float.
  Return null if not mentioned.

- employment_type: "private", "government", or "contract".
  Return null if not mentioned.

- confidence: your confidence in the extraction, from 0.0 to 1.0.
  1.0 = clear employment law query, 0.5 = ambiguous, 0.2 = unclear

- is_valid: true if the query is about Indian employment or labour law.
  false if the query is completely unrelated to employment law.

- validation_note: if is_valid is false, explain why in one sentence.
  Empty string if is_valid is true.

Example:
Query: "I was fired without notice after 3 years at a private company in Karnataka"
Response:
{
  "concepts": ["wrongful termination", "notice period"],
  "jurisdiction": "Karnataka",
  "entity_type": "employee",
  "years_of_service": 3.0,
  "employment_type": "private",
  "confidence": 1.0,
  "is_valid": true,
  "validation_note": ""
}

Return ONLY valid JSON. No markdown. No explanation. No code fences.
"""


def build_extraction_prompt(raw_query: str) -> str:
    """
    Build the user-turn prompt sent to Gemini.

    Parameters:
        raw_query: Original user query.

    Returns:
        Formatted query string for the LLM.
    """
    return f"Query: {raw_query}"


def build_adk_payload(raw_query: str, session_id: str, user_id: str) -> dict:
    """
    Build the complete ADK-compatible payload.

    Parameters:
        raw_query: Original user query.
        session_id: Session identifier.
        user_id: User identifier.

    Returns:
        ADK payload dictionary ready for runner invocation.
    """
    return {
        "app_name": "legal_graph_rag",
        "user_id": f"u_{user_id}",
        "session_id": f"s_{session_id}",
        "new_message": {
            "role": "user",
            "parts": [
                {
                    "text": build_extraction_prompt(raw_query),
                }
            ],
        },
    }


def parse_gemini_response(response_text: str) -> dict:
    """
    Parse Gemini raw response text into a dictionary.

    Parameters:
        response_text: Raw text returned by Gemini.

    Returns:
        Parsed response dictionary. Never raises.
    """
    fallback = {
        "concepts": [],
        "jurisdiction": "Central",
        "entity_type": "employee",
        "years_of_service": None,
        "employment_type": None,
        "confidence": 0.0,
        "is_valid": False,
        "validation_note": "Failed to parse LLM response.",
    }

    try:
        parsed = json.loads(response_text.strip())
        logger.debug("Parsed Gemini response using direct JSON parse.")
        return parsed
    except Exception:
        pass

    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        parsed = json.loads(cleaned.strip())
        logger.warning("Parsed Gemini response after stripping markdown fences.")
        return parsed
    except Exception:
        pass

    try:
        start = response_text.find("{")
        end = response_text.rfind("}")

        if start != -1 and end != -1 and start < end:
            json_substring = response_text[start : end + 1]
            parsed = json.loads(json_substring)
            logger.warning("Parsed Gemini response by extracting JSON substring.")
            return parsed
    except Exception:
        pass

    logger.error("Failed to parse Gemini response. Returning fallback extraction.")
    return fallback


def to_extracted_entities(parsed_dict: dict, raw_query: str) -> ExtractedEntities:
    """
    Map a parsed dictionary to an ExtractedEntities dataclass.

    Parameters:
        parsed_dict: Parsed Gemini response dictionary.
        raw_query: Original user query.

    Returns:
        ExtractedEntities instance. Never raises.
    """
    try:
        return ExtractedEntities(
            raw_query=raw_query,
            concepts=parsed_dict.get("concepts", []),
            jurisdiction=parsed_dict.get("jurisdiction", "Central"),
            entity_type=parsed_dict.get("entity_type", "employee"),
            years_of_service=parsed_dict.get("years_of_service"),
            employment_type=parsed_dict.get("employment_type"),
            confidence=parsed_dict.get("confidence", 0.0),
            is_valid=parsed_dict.get("is_valid", False),
            validation_note=parsed_dict.get("validation_note", ""),
        )
    except Exception:
        logger.error("Failed to map parsed response to ExtractedEntities.")
        return ExtractedEntities(raw_query=raw_query)


if __name__ == "__main__":
    import json

    test_query = "I was fired without notice after 3 years at a private company in Karnataka"

    print("=== build_extraction_prompt ===")
    print(build_extraction_prompt(test_query))

    print("\n=== build_adk_payload ===")
    payload = build_adk_payload(test_query, session_id="test123", user_id="dev")
    print(json.dumps(payload, indent=2))

    print("\n=== parse_gemini_response — clean JSON ===")
    clean = '{"concepts": ["wrongful termination", "notice period"], "jurisdiction": "Karnataka", "entity_type": "employee", "years_of_service": 3.0, "employment_type": "private", "confidence": 1.0, "is_valid": true, "validation_note": ""}'
    parsed = parse_gemini_response(clean)
    print(parsed)

    print("\n=== parse_gemini_response — markdown wrapped ===")
    wrapped = f"```json\n{clean}\n```"
    parsed2 = parse_gemini_response(wrapped)
    print(parsed2)

    print("\n=== to_extracted_entities ===")
    entities = to_extracted_entities(parsed, test_query)
    print(entities)