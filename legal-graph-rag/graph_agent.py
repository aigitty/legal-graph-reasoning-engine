"""
graph_agent.py

Builds and compiles the LangGraph workflow for the Legal Graph RAG agent layer.

STEP 7 (current):
    entity_extraction -> concept_grounding -> graph_traversal
        -> sufficiency_evaluator -> [conditional]
             "expand" -> graph_expansion -> graph_traversal   (loop)
             "end"    -> answer_synthesis -> output_guardrail
                            -> final_response -> END

answer_synthesis is the THIRD and final Gemini call. After it, two DETERMINISTIC
nodes close the pipeline:
  - output_guardrail: verifies citations against Neo4j AND the retrieval set,
    strips any that fail, computes the confidence score, sets status, and
    injects the disclaimer (CLAUDE.md rules 3 & 5, section 6).
  - final_response: pure formatting — assembles the user-facing final_answer for
    every terminal status (ok / insufficient_evidence / out_of_domain /
    rejected / error). It is the safety net and never raises.

Loop termination is governed by MAX_ITERATIONS in
agents.nodes.sufficiency_node, checked deterministically before any Gemini
call — see _route_after_sufficiency.

Public surface stays stable:
    build_graph() -> compiled graph
    run(raw_query) -> LegalQueryState
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END

from agents.graph_state import LegalQueryState
from agents.nodes.extraction_node import entity_extraction_node
from agents.nodes.grounding_node import concept_grounding_node
from agents.nodes.retrieval_node import graph_traversal_node
from agents.nodes.sufficiency_node import sufficiency_evaluator_node, MAX_ITERATIONS
from agents.nodes.expansion_node import graph_expansion_node
from agents.nodes.synthesis_node import answer_synthesis_node
from agents.nodes.output_guardrail_node import output_guardrail_node
from agents.nodes.final_response_node import final_response_node
from config import cfg


def _route_after_sufficiency(state: LegalQueryState) -> str:
    if state.sufficiency is None or state.sufficiency.sufficient:
        return "end"
    if state.retrieval_iterations >= MAX_ITERATIONS:
        return "end"
    if state.retrieval is None or state.retrieval.is_empty:
        return "end"  # nothing to expand from — grounding produced no seeds
    return "expand"


def build_graph():
    builder = StateGraph(LegalQueryState)
    builder.add_node("entity_extraction", entity_extraction_node)
    builder.add_node("concept_grounding", concept_grounding_node)
    builder.add_node("graph_traversal", graph_traversal_node)
    builder.add_node("sufficiency_evaluator", sufficiency_evaluator_node)
    builder.add_node("graph_expansion", graph_expansion_node)
    builder.add_node("answer_synthesis", answer_synthesis_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("final_response", final_response_node)

    builder.add_edge(START, "entity_extraction")
    builder.add_edge("entity_extraction", "concept_grounding")
    builder.add_edge("concept_grounding", "graph_traversal")
    builder.add_edge("graph_traversal", "sufficiency_evaluator")

    builder.add_conditional_edges(
        "sufficiency_evaluator",
        _route_after_sufficiency,
        {"expand": "graph_expansion", "end": "answer_synthesis"},
    )
    builder.add_edge("graph_expansion", "graph_traversal")
    builder.add_edge("answer_synthesis", "output_guardrail")
    builder.add_edge("output_guardrail", "final_response")
    builder.add_edge("final_response", END)

    return builder.compile()


GRAPH = build_graph()


def run(raw_query: str) -> LegalQueryState:
    out = GRAPH.invoke(
        {"raw_query": raw_query},
        config={"recursion_limit": cfg.LANGGRAPH_RECURSION_LIMIT},
    )
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

    print(f"max_hops: {state.max_hops}  |  retrieval_iterations: {state.retrieval_iterations}")

    print(f"Cited section_ids:    {state.cited_section_ids}")
    print(f"Verified section_ids: {state.verified_section_ids}")
    print(f"Status:     {state.status}")
    print(f"Confidence: {state.confidence}  factors={state.confidence_factors}")

    if state.final_answer:
        print("-" * 80)
        print("FINAL ANSWER:")
        print(state.final_answer)

    print("=" * 80)
    print()


if __name__ == "__main__":
    print("Legal Graph RAG — Indian employment law reasoning engine")
    print("Type your question, or 'exit'/'quit' to leave.\n")
    while True:
        try:
            raw_query = input("Enter your legal query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if raw_query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if not raw_query:
            print("Please enter a query (or type 'exit' to quit).\n")
            continue

        _print_result(raw_query, run(raw_query))