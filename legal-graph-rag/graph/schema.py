"""
graph/schema.py

Single source of truth for the legal knowledge graph structure.
Every other file (ingest, traversal, agents) imports from here.

MVP data coverage:
  Nodes:         Act, Section, Concept, Ruling (Ruling = v2, defined now)
  Relationships: HAS_SECTION, CITES, OVERRIDES, APPLIES_TO

Not in MVP (v2):
  IMPOSES, PROTECTS — require penalty extractor, not built yet
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────
# NODE LABELS
# Exact strings used in Neo4j.
# Always reference via NodeLabel.X — never hardcode the string.
# ─────────────────────────────────────────────────────────────

class NodeLabel(str, Enum):
    ACT     = "Act"      # full statute e.g. Industrial Disputes Act 1947
    SECTION = "Section"  # one section within an act e.g. Section 25F
    CONCEPT = "Concept"  # plain-language concept e.g. "wrongful termination"
    RULING  = "Ruling"   # court ruling (v2 — no data yet, schema defined now)


# ─────────────────────────────────────────────────────────────
# RELATIONSHIP TYPES
# Exact strings used in Neo4j.
# MVP active: HAS_SECTION, CITES, OVERRIDES, APPLIES_TO
# v2: IMPOSES, PROTECTS (removed for now — no data)
# ─────────────────────────────────────────────────────────────

class RelType(str, Enum):
    # structural
    HAS_SECTION = "HAS_SECTION"  # Act → Section

    # reasoning edges — populated by relationship_extractor.py
    CITES       = "CITES"        # Section → Section (cross-reference within act)
    OVERRIDES   = "OVERRIDES"    # Act → Act (newer law supersedes older)

    # populated by ontology_loader.py
    APPLIES_TO  = "APPLIES_TO"   # Section → Concept

    # v2 — populated by ruling ingest (Layer 2)
    INTERPRETS  = "INTERPRETS"   # Ruling → Section


# ─────────────────────────────────────────────────────────────
# NODE DATACLASSES
#
# Field names EXACTLY match the keys in sections.jsonl
# so graph_builder.py can pass records directly without remapping.
#
# sections.jsonl keys:
#   act_id, act_name, year, jurisdiction, short_name,
#   section_id, section_number, section_title, section_text, source_file
# ─────────────────────────────────────────────────────────────

@dataclass
class ActNode:
    """
    One per unique act_id in sections.jsonl.
    Created once, shared by all sections belonging to this act.

    Example:
        act_id       = "IDA_1947"
        act_name     = "Industrial Disputes Act"
        year         = 1947
        jurisdiction = "Central"
        short_name   = "IDA"
    """
    act_id:      str
    act_name:    str          # matches sections.jsonl field name
    year:        int
    jurisdiction: str         # "Central" or state name e.g. "Karnataka"
    short_name:  str          # "IDA" — compact identifier used in section_ids


@dataclass
class SectionNode:
    """
    One per line in sections.jsonl.

    Field names match sections.jsonl exactly:
        section_number, section_title, section_text, section_id

    Example:
        section_id     = "IDA_1947_S25F"
        section_number = "25F"
        section_title  = "Conditions precedent to retrenchment of workmen"
        section_text   = "No workman employed in any industry..."
        act_id         = "IDA_1947"
        source_file    = "industrial_disputes_act_1947.pdf"
    """
    section_id:     str   # unique identifier — used as Neo4j merge key
    section_number: str   # "25F" — matches jsonl field name
    section_title:  str   # matches jsonl field name
    section_text:   str   # matches jsonl field name
    act_id:         str   # foreign key → ActNode.act_id
    source_file:    str   # which PDF this came from — for debugging


@dataclass
class ConceptNode:
    """
    Plain-language legal concept — loaded from data/ontology/concept_map.json
    by ontology_loader.py. Maps user query terms to Section nodes.

    Example:
        concept_id  = "wrongful_termination"
        name        = "wrongful termination"
        description = "Termination of employment without following legal procedure"
        aliases     = ["fired without cause", "illegal termination", "unfair dismissal"]
    """
    concept_id:  str
    name:        str
    description: str
    aliases:     List[str] = field(default_factory=list)


@dataclass
class RulingNode:
    """
    Court ruling — v2, loaded from Indian Kanoon / Kaggle data.
    Schema defined now so graph_builder.py v2 knows the contract.
    No data in MVP.

    Example:
        ruling_id   = "SC_1992_WorkmenVsMeenakshiMills"
        case_name   = "Workmen v. Meenakshi Mills"
        court       = "Supreme Court"
        year        = 1992
        citation    = "1994 AIR 2696"
        summary     = "Section 25F applies to all retrenched workmen..."
    """
    ruling_id:  str
    case_name:  str
    court:      str
    year:       int
    citation:   Optional[str] = None
    summary:    Optional[str] = None


# ─────────────────────────────────────────────────────────────
# RELATIONSHIP DATACLASS
# Used by graph_builder.py to load relationships.jsonl into Neo4j.
#
# relationships.jsonl keys (from relationship_extractor.py):
#   source_section_id, source_act_id, relationship_type,
#   target_section_id, target_section_number, evidence_text
# ─────────────────────────────────────────────────────────────

@dataclass
class Relationship:
    """
    A directed edge between two nodes.

    source_section_id and target_section_id reference SectionNode.section_id.
    rel_type must be a RelType enum value.
    properties holds any extra edge metadata (e.g. evidence_text for CITES).
    """
    from_id:    str
    to_id:      str
    rel_type:   RelType
    properties: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# HELPERS — used by graph_builder.py and queries.py
# ─────────────────────────────────────────────────────────────

def label(node_label: NodeLabel) -> str:
    """Returns the Neo4j label string. Use instead of hardcoding."""
    return node_label.value


def rel(rel_type: RelType) -> str:
    """Returns the Neo4j relationship type string."""
    return rel_type.value


# ─────────────────────────────────────────────────────────────
# SCHEMA SUMMARY — run directly to verify schema loads cleanly
# python graph/schema.py
# ─────────────────────────────────────────────────────────────

def print_schema_summary():
    print("=" * 50)
    print("Legal Graph — MVP Schema")
    print("=" * 50)
    print("\nNode types:")
    for n in NodeLabel:
        note = " (v2 — no data yet)" if n == NodeLabel.RULING else ""
        print(f"  ({n.value}){note}")
    print("\nRelationship types:")
    rel_notes = {
        "HAS_SECTION": "Act → Section",
        "CITES":       "Section → Section  [relationship_extractor.py]",
        "OVERRIDES":   "Act → Act          [manual / v2]",
        "APPLIES_TO":  "Section → Concept  [ontology_loader.py]",
        "INTERPRETS":  "Ruling → Section   [v2 — no data yet]",
    }
    for r in RelType:
        print(f"  [{r.value}]  {rel_notes.get(r.value, '')}")
    print("\nSchema loaded successfully.")


if __name__ == "__main__":
    print_schema_summary()