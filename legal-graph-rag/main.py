from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from graph.queries import test_connection
from graph.traversal import TraversalResult, traverse


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ACT_NAMES: dict[str, str] = {
    "IDA_1947":   "Industrial Disputes Act, 1947",
    "PGA_1972":   "Payment of Gratuity Act, 1972",
    "MWA_1948":   "Minimum Wages Act, 1948",
    "KSEA_1961":  "Karnataka Shops and Establishments Act, 1961",
    "COW_2019":   "Code on Wages, 2019",
}


def _confidence_label(confidence: float) -> str:
    if confidence >= 1.0:
        return "HIGH"
    if confidence >= 0.6:
        return "MEDIUM"
    return "LOW"


def _section_id(section: dict) -> str:
    return str(section.get("section_id") or section.get("id") or "UNKNOWN")


def _act_name(section: dict) -> str:
    act_id = str(section.get("act_id") or "")
    return ACT_NAMES.get(act_id, act_id)


def _section_number(section: dict) -> str:
    return str(section.get("section_number") or "N/A")


def _section_title(section: dict) -> str:
    return str(section.get("section_title") or section.get("title") or "Untitled section")


def _section_text(section: dict) -> str:
    return str(section.get("section_text") or section.get("text") or "")


def _preview_text(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars]
    last_space = trimmed.rfind(" ")

    if last_space > 0:
        trimmed = trimmed[:last_space]

    return trimmed.rstrip() + "..."


def format_result(result: TraversalResult) -> str:
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("LEGAL GRAPH TRAVERSAL RESULT")
    lines.append("=" * 80)
    lines.append(f"Query: {result.concept_queried}")
    lines.append(
        f"Confidence: {result.confidence} ({_confidence_label(result.confidence)})"
    )
    lines.append(f"Hops taken: {result.hops_taken}")

    if result.is_empty:
        lines.append("─" * 80)
        lines.append("No matching legal sections were found.")
        lines.append("")
        lines.append("Try queries like:")
        lines.append("  - wrongful termination")
        lines.append("  - gratuity")
        lines.append("  - minimum wages")
        lines.append("  - salary not paid")
        lines.append("=" * 80)
        return "\n".join(lines)

    sections_by_act: dict[str, list[dict]] = defaultdict(list)
    for section in result.all_sections:
        sections_by_act[_act_name(section)].append(section)

    lines.append("─" * 80)
    lines.append(f"Total sections found: {len(result.all_sections)}")
    lines.append("Sections by Act:")

    for act, sections in sections_by_act.items():
        lines.append(f"  - {act}: {len(sections)}")

    lines.append("─" * 80)
    lines.append("SECTIONS")
    lines.append("─" * 80)

    for act, sections in sections_by_act.items():
        lines.append("")
        lines.append(f"[{act}]")
        lines.append("-" * 80)

        for index, section in enumerate(sections, start=1):
            text_preview = _preview_text(_section_text(section), max_chars=300)

            lines.append(f"{index}. Section ID: {_section_id(section)}")
            lines.append(f"   Act: {_act_name(section)}")
            lines.append(f"   Section: {_section_number(section)}")
            lines.append(f"   Title: {_section_title(section)}")
            lines.append(f"   Text: {text_preview}")
            lines.append("")

    lines.append("─" * 80)
    lines.append("CITES EDGES")
    lines.append("─" * 80)

    if not result.cites_edges:
        lines.append("No CITES edges found between returned sections.")
    else:
        for edge in result.cites_edges:
            source = edge.get("source_section_id") or edge.get("source_id")
            target = edge.get("target_section_id") or edge.get("target_id")
            lines.append(f"{source} → cites → {target}")

    execution_time_ms = getattr(result, "execution_time_ms", None)
    if execution_time_ms is not None:
        lines.append("─" * 80)
        lines.append(f"Execution time: {execution_time_ms:.2f} ms")

    lines.append("=" * 80)
    return "\n".join(lines)


def run_query(query: str) -> TraversalResult | None:
    query = query.strip()

    if not query:
        print("Please enter a non-empty legal query.")
        return None

    try:
        start_time = time.perf_counter()
        result = traverse(
            concept_name=query.lower(),
            max_hops=2,
            max_sections=15,
        )
        end_time = time.perf_counter()

        result.execution_time_ms = (end_time - start_time) * 1000
        return result

    except Exception as exc:
        logger.exception("Traversal failed")
        print("=" * 80)
        print("ERROR")
        print("=" * 80)
        print("Something went wrong while traversing the graph.")
        print(f"Details: {exc}")
        print("=" * 80)
        return None


def print_banner() -> None:
    print("=" * 80)
    print("LEGAL CASE REASONING ENGINE — GRAPH RAG RUNTIME")
    print("=" * 80)
    print("Step 7 runtime: terminal-based Neo4j graph traversal only.")
    print("No LLM. No agents. No guardrails.")
    print("")
    print("Try queries like:")
    print("  - wrongful termination")
    print("  - gratuity")
    print("  - minimum wages")
    print("  - salary not paid")
    print("")
    print("Type 'exit', 'quit', or 'q' to close.")
    print("=" * 80)


def main() -> None:
    print_banner()

    print("Checking Neo4j connection...")
    try:
        test_connection()

    except Exception as exc:
        print("=" * 80)
        print("Neo4j connection failed.")
        print("Please check your .env file and make sure Neo4j is running.")
        print(f"Details: {exc}")
        print("=" * 80)
        sys.exit(1)

    print("Neo4j connection successful.")

    while True:
        print("")
        query = input("legal-graph-rag> ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye. Legal Graph RAG runtime closed.")
            break

        result = run_query(query)
        if result is not None:
            print(format_result(result))


if __name__ == "__main__":
    main()