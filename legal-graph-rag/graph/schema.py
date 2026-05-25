"""
graph/schema.py

Defines all node types, edge types, and their properties
for the legal knowledge graph stored in Neo4j.

This file is the single source of truth for the graph structure.
Every other file (ingest, traversal, agents) imports from here.
Nothing in this file talks to Neo4j — it is pure definitions only.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────
# NODE LABELS
# These are the exact strings used in Neo4j
# e.g. CREATE (n:Section {number: "25F"})
# ─────────────────────────────────────────────

class NodeLabel(str, Enum):
    ACT      = "Act"        # a full statute e.g. Industrial Disputes Act 1947
    SECTION  = "Section"    # one section within an act e.g. Section 25F
    CONCEPT  = "Concept"    # plain-language legal concept e.g. "wrongful termination"
    PENALTY  = "Penalty"    # punishment defined by a section
    ENTITY   = "Entity"     # a party type e.g. employer, employee, consumer


# ─────────────────────────────────────────────
# RELATIONSHIP TYPES
# These are the exact strings used in Neo4j
# e.g. (section)-[:CITES]->(other_section)
# ─────────────────────────────────────────────

class RelType(str, Enum):
    # structural
    HAS_SECTION  = "HAS_SECTION"   # Act → Section
    # legal reasoning edges
    CITES        = "CITES"         # Section → Section (cross-reference)
    OVERRIDES    = "OVERRIDES"     # Act → Act (newer law supersedes older)
    APPLIES_TO   = "APPLIES_TO"    # Section → Concept
    IMPOSES      = "IMPOSES"       # Section → Penalty
    PROTECTS     = "PROTECTS"      # Section → Entity


# ─────────────────────────────────────────────
# NODE DATACLASSES
# Used by pdf_parser.py and graph_builder.py
# to create structured objects before loading
# into Neo4j. Also used for type hints in agents.
# ─────────────────────────────────────────────

@dataclass
class ActNode:
    """
    Represents a full Indian statute.
    Example: Industrial Disputes Act, 1947
    """
    name: str                          # "Industrial Disputes Act"
    year: int                          # 1947
    jurisdiction: str                  # "Central" or state name e.g. "Karnataka"
    short_name: str                    # "IDA" — used as a compact identifier
    full_text: Optional[str] = None    # raw text of the full act (optional, large)

    def node_id(self) -> str:
        """Unique identifier for this node in Neo4j."""
        return f"{self.short_name}_{self.year}"


@dataclass
class SectionNode:
    """
    Represents one section within an act.
    Example: Section 25F of the Industrial Disputes Act 1947
    """
    number: str          # "25F" — section number as string (can contain letters)
    title: str           # "Conditions precedent to retrenchment of workmen"
    text: str            # full text of this section
    act_id: str          # references ActNode.node_id() e.g. "IDA_1947"

    def node_id(self) -> str:
        """Unique identifier — act + section number."""
        return f"{self.act_id}_S{self.number}"


@dataclass
class ConceptNode:
    """
    A plain-language legal concept that maps user queries to graph nodes.
    Loaded from data/ontology/concept_map.json.
    Example: "wrongful termination" → maps to Section 25F IDA 1947
    """
    name: str                    # "wrongful termination"
    description: str             # brief explanation of the concept
    aliases: List[str] = field(default_factory=list)
    # e.g. ["fired without cause", "illegal termination", "unfair dismissal"]

    def node_id(self) -> str:
        return self.name.lower().replace(" ", "_")


@dataclass
class PenaltyNode:
    """
    A penalty or remedy defined by a section.
    Example: 15 days wages per completed year of service (Section 25F)
    """
    penalty_type: str             # "compensation", "imprisonment", "fine"
    description: str              # human-readable penalty description
    min_amount: Optional[str] = None   # "15 days wages" — string because formula
    max_amount: Optional[str] = None
    imprisonment: Optional[str] = None  # e.g. "up to 3 months"
    section_id: str = ""          # references SectionNode.node_id()

    def node_id(self) -> str:
        return f"penalty_{self.section_id}"


@dataclass
class EntityNode:
    """
    A party type that a section applies to or protects.
    Example: "workman", "employer", "consumer"
    """
    name: str           # "workman"
    entity_type: str    # "employee", "employer", "consumer", "government"
    description: str

    def node_id(self) -> str:
        return self.name.lower().replace(" ", "_")


# ─────────────────────────────────────────────
# RELATIONSHIP DATACLASSES
# Used by pdf_parser.py to emit structured
# edges into relationships.jsonl before
# graph_builder.py loads them into Neo4j.
# ─────────────────────────────────────────────

@dataclass
class Relationship:
    """
    A generic directed edge between two nodes.
    from_id and to_id reference node_id() values.
    rel_type must be a RelType enum value.
    properties holds any extra edge metadata.
    """
    from_id: str
    to_id: str
    rel_type: RelType
    properties: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# CYPHER LABEL HELPERS
# Convenience functions used in queries.py
# and graph_builder.py to avoid hardcoding
# label strings throughout the codebase.
# ─────────────────────────────────────────────

def label(node_label: NodeLabel) -> str:
    """Returns the Neo4j label string for a node type."""
    return node_label.value


def rel(rel_type: RelType) -> str:
    """Returns the Neo4j relationship type string."""
    return rel_type.value


# ─────────────────────────────────────────────
# SCHEMA SUMMARY (printed during ingest for
# confirmation that schema loaded correctly)
# ─────────────────────────────────────────────

def print_schema_summary():
    print("=== Legal Graph Schema ===")
    print("\nNode types:")
    for n in NodeLabel:
        print(f"  ({n.value})")
    print("\nRelationship types:")
    for r in RelType:
        print(f"  [{r.value}]")
    print("\nSchema loaded successfully.")


if __name__ == "__main__":
    print_schema_summary()