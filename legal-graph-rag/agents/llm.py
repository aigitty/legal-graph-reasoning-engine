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

from config import cfg

DEFAULT_MODEL = cfg.GEMINI_MODEL


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    thinking_budget: int | None = None,
) -> ChatVertexAI:
    """
    thinking_budget: optional, Gemini 2.5 "thinking" token budget. Left
    unset (None) by default -> existing callers (extraction_node,
    sufficiency_node) get exactly the same ChatVertexAI as before.

    Pass 0 to disable extended thinking entirely. On Gemini 2.5 models,
    thinking tokens are drawn from the same max_output_tokens budget as the
    visible response, and the thinking budget is dynamic by default — for
    tasks that don't need multi-step reasoning (e.g. answer synthesis, which
    is mostly constrained transformation + formatting), this can silently
    eat most of max_output_tokens and truncate the visible answer. Setting
    thinking_budget=0 makes the remaining budget fully available to the
    visible output.
    """
    project = cfg.GOOGLE_CLOUD_PROJECT or os.environ["GOOGLE_CLOUD_PROJECT"]
    location = cfg.GOOGLE_CLOUD_LOCATION

    kwargs: dict = dict(
        model=model,
        project=project,
        location=location,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget

    return ChatVertexAI(**kwargs)


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