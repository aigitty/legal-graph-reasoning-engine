"""
graph_agent.py

Builds and compiles the LangGraph workflow for the Legal Graph RAG agent layer.

STEP 4 (current):
    entity_extraction -> concept_grounding -> graph_traversal -> sufficiency_evaluator

sufficiency_evaluator is the second Gemini call (structured output).
NO LOOP YET — this step only observes verdicts on real queries. The
expansion loop + conditional edge is the next increment.

Public surface stays stable:
    build_graph() -> compiled graph
    run(raw_query) -> LegalQueryState
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # load Neo4j/Aura + GCP env vars before any client construction

from langgraph.graph import StateGraph, START, END

from agents.graph_state import LegalQueryState
from agents.nodes.extraction_node import entity_extraction_node
from agents.nodes.grounding_node import concept_grounding_node
from agents.nodes.retrieval_node import graph_traversal_node
from agents.nodes.sufficiency_node import sufficiency_evaluator_node


def build_graph():
    builder = StateGraph(LegalQueryState)
    builder.add_node("entity_extraction", entity_extraction_node)
    builder.add_node("concept_grounding", concept_grounding_node)
    builder.add_node("graph_traversal", graph_traversal_node)
    builder.add_node("sufficiency_evaluator", sufficiency_evaluator_node)

    builder.add_edge(START, "entity_extraction")
    builder.add_edge("entity_extraction", "concept_grounding")
    builder.add_edge("concept_grounding", "graph_traversal")
    builder.add_edge("graph_traversal", "sufficiency_evaluator")
    builder.add_edge("sufficiency_evaluator", END)
    return builder.compile()


GRAPH = build_graph()


def run(raw_query: str) -> LegalQueryState:
    out = GRAPH.invoke({"raw_query": raw_query})
    if isinstance(out, LegalQueryState):
        return out
    return LegalQueryState(**out)


def _print_result(label: str, state: LegalQueryState) -> None:
    print("=" * 80)
    print(f"QUERY: {label}")
    print("=" * 80)

    if state.extraction:
        e = state.extraction
        print(f"legal_concepts:   {e.legal_concepts}")
        print(f"jurisdiction:     {e.jurisdiction}")
        print(f"in_domain:        {e.in_domain}")

    print(f"Grounded concepts: {state.grounded_concepts}")
    if state.warnings:
        print(f"Warnings: {state.warnings}")

    if state.error:
        print(f"ERROR: {state.error}")
    elif state.retrieval is None:
        print("No retrieval produced.")
    else:
        r = state.retrieval
        print(f"Total sections found: {r.total_found}")
        print(f"Primary sections: {[s.section_id for s in r.primary_sections]}")

    if state.sufficiency:
        print(f"Sufficient: {state.sufficiency.sufficient}")
        if state.sufficiency.missing:
            print(f"Missing:    {state.sufficiency.missing}")
    print(f"Retrieval iterations: {state.retrieval_iterations}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    _print_result(
        "gratuity",
        run("gratuity"),
    )

    _print_result(
        "I was fired without notice after 3 years at a private company in Karnataka",
        run("I was fired without notice after 3 years at a private company in Karnataka"),
    )

    _print_result(
        "What is the procedure for filing an RTI request?",
        run("What is the procedure for filing an RTI request?"),
    )