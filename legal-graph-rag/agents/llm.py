"""
agents/llm.py

Gemini via Vertex AI client factory.

Auth: Application Default Credentials (ADC) only — via
`gcloud auth application-default login`. Do NOT set
GOOGLE_APPLICATION_CREDENTIALS; ChatVertexAI calls google.auth.default()
internally and discovers the ADC file automatically. The ADC quota project
(already configured) covers billing/quota for Vertex AI calls.

Required env vars (already present in .env):
    GOOGLE_CLOUD_PROJECT
    GOOGLE_CLOUD_LOCATION

DEPENDENCY NOTE — ChatVertexAI deprecation:
    langchain-google-vertexai >=3.2 deprecates ChatVertexAI in favor of
    ChatGoogleGenerativeAI(vertexai=True) from langchain-google-genai.
    We are deliberately staying on ChatVertexAI for this MVP:
      - it works correctly (verified via this module's smoke test)
      - the recommended replacement has reported 50-90% latency
        regressions and active auth bugs for project/ADC-based Vertex
        setups (as of late 2025 / early 2026)
    requirements.txt MUST pin: langchain-google-vertexai<4.0.0
    so ChatVertexAI remains available. Revisit this migration once the
    ChatGoogleGenerativeAI Vertex path stabilizes (tracked as a v2 item).
"""

from __future__ import annotations

import os
import warnings

# Suppress only the two known ChatVertexAI deprecation messages — not all
# warnings — so real issues stay visible.
warnings.filterwarnings("ignore", message=r"Use `ChatGoogleGenerativeAI`.*")
warnings.filterwarnings("ignore", message=r"The class `ChatVertexAI` was deprecated.*")

from langchain_google_vertexai import ChatVertexAI

DEFAULT_MODEL = "gemini-2.5-flash"


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
) -> ChatVertexAI:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    return ChatVertexAI(
        model=model,
        project=project,
        location=location,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 80)
    print("VERTEX AI / ADC AUTH SMOKE TEST")
    print(f"Project:  {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    print(f"Location: {os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')}")
    print(f"Model:    {DEFAULT_MODEL}")
    print("=" * 80)

    llm = get_llm()
    response = llm.invoke("Reply with exactly one word: OK")
    print(f"Response: {response.content!r}")
    print("=" * 80)