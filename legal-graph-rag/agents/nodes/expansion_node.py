"""
agents/nodes/expansion_node.py

Deterministic graph-expansion node (architecture section 7).

Single MVP strategy: increase traversal depth (max_hops) by one hop,
capped at MAX_HOPS_CAP. Looping back to graph_traversal happens via the
conditional edge in graph_agent.py, not here. Always deterministic.
"""

from __future__ import annotations

from agents.graph_state import LegalQueryState
from config import cfg

MAX_HOPS_CAP = cfg.MAX_HOPS_CAP


def graph_expansion_node(state: LegalQueryState) -> dict:
    new_hops = min(state.max_hops + 1, MAX_HOPS_CAP)

    if new_hops == state.max_hops:
        return {
            "warnings": state.warnings + [
                f"Cannot expand further: max_hops cap ({MAX_HOPS_CAP}) reached."
            ]
        }

    missing = state.sufficiency.missing if state.sufficiency else "unspecified"
    return {
        "max_hops": new_hops,
        "warnings": state.warnings + [
            f"Expanding traversal depth {state.max_hops} -> {new_hops} "
            f"(sufficiency gap: {missing!r})."
        ],
    }