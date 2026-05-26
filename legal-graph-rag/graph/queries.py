"""
Neo4j query access layer for the Legal Graph RAG project.

This module is the single source of raw Cypher queries in the codebase.
Other modules should call functions from this file instead of writing Cypher directly.
"""

from typing import Any, Dict, List, Optional

from graph.db_connection import get_driver
from graph.schema import NodeLabel, RelType


# ---------------------------------------------------------------------------
# Indexes / Constraints
# ---------------------------------------------------------------------------

def ensure_indexes() -> None:
    """
    Create indexes/constraints needed for fast graph loading and traversal.

    Parameters:
        None

    Returns:
        None
    """
    driver = get_driver()

    section_label = NodeLabel.SECTION.value
    act_label = NodeLabel.ACT.value
    concept_label = NodeLabel.CONCEPT.value

    queries = [
        f"""
        CREATE CONSTRAINT section_id_unique IF NOT EXISTS
        FOR (s:{section_label})
        REQUIRE s.section_id IS UNIQUE
        """,
        f"""
        CREATE CONSTRAINT act_id_unique IF NOT EXISTS
        FOR (a:{act_label})
        REQUIRE a.act_id IS UNIQUE
        """,
        f"""
        CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
        FOR (c:{concept_label})
        REQUIRE c.concept_id IS UNIQUE
        """,
        f"""
        CREATE INDEX concept_name_index IF NOT EXISTS
        FOR (c:{concept_label})
        ON (c.name)
        """,
    ]

    with driver.session() as session:
        for query in queries:
            session.run(query)


# ---------------------------------------------------------------------------
# Write Operations
# ---------------------------------------------------------------------------

def create_act(
    tx: Any,
    act_id: str,
    act_name: str,
    year: int,
    jurisdiction: str,
    short_name: str,
) -> None:
    """
    Create or update an Act node.

    Parameters:
        tx: Active Neo4j transaction object.
        act_id: Stable unique identifier for the Act.
        act_name: Full name of the Act.
        year: Year of enactment.
        jurisdiction: Jurisdiction where the Act applies.
        short_name: Short readable name for the Act.

    Returns:
        None
    """
    act_label = NodeLabel.ACT.value

    query = f"""
    MERGE (a:{act_label} {{act_id: $act_id}})
    SET
        a.act_name = $act_name,
        a.year = $year,
        a.jurisdiction = $jurisdiction,
        a.short_name = $short_name
    """

    tx.run(
        query,
        act_id=act_id,
        act_name=act_name,
        year=year,
        jurisdiction=jurisdiction,
        short_name=short_name,
    )


def create_section(
    tx: Any,
    section_id: str,
    section_number: str,
    section_title: str,
    section_text: str,
    act_id: str,
    source_file: str,
) -> None:
    """
    Create or update a Section node and connect it to its Act using HAS_SECTION.

    Parameters:
        tx: Active Neo4j transaction object.
        section_id: Stable unique identifier for the section.
        section_number: Legal section number.
        section_title: Title/headline of the section.
        section_text: Full text of the legal section.
        act_id: ID of the Act this section belongs to.
        source_file: Source PDF filename.

    Returns:
        None
    """
    section_label = NodeLabel.SECTION.value
    act_label = NodeLabel.ACT.value
    has_section_rel = RelType.HAS_SECTION.value

    query = f"""
    MATCH (a:{act_label} {{act_id: $act_id}})
    MERGE (s:{section_label} {{section_id: $section_id}})
    SET
        s.section_number = $section_number,
        s.section_title = $section_title,
        s.section_text = $section_text,
        s.act_id = $act_id,
        s.source_file = $source_file
    MERGE (a)-[:{has_section_rel}]->(s)
    """

    tx.run(
        query,
        section_id=section_id,
        section_number=section_number,
        section_title=section_title,
        section_text=section_text,
        act_id=act_id,
        source_file=source_file,
    )


def create_concept(
    tx: Any,
    concept_id: str,
    name: str,
    description: str,
    aliases: List[str],
) -> None:
    """
    Create or update a Concept node.

    Parameters:
        tx: Active Neo4j transaction object.
        concept_id: Stable unique identifier for the concept.
        name: Human-readable concept name.
        description: Description of what the concept means.
        aliases: List of natural-language phrases that map to this concept.

    Returns:
        None
    """
    concept_label = NodeLabel.CONCEPT.value

    query = f"""
    MERGE (c:{concept_label} {{concept_id: $concept_id}})
    SET
        c.name = $name,
        c.description = $description,
        c.aliases = $aliases
    """

    tx.run(
        query,
        concept_id=concept_id,
        name=name,
        description=description,
        aliases=aliases,
    )


def create_applies_to(
    tx: Any,
    section_id: str,
    concept_id: str,
    relevance: str,
) -> None:
    """
    Create an APPLIES_TO relationship from Section to Concept.

    Parameters:
        tx: Active Neo4j transaction object.
        section_id: Source Section node ID.
        concept_id: Target Concept node ID.
        relevance: Relevance label such as primary or supporting.

    Returns:
        None
    """
    section_label = NodeLabel.SECTION.value
    concept_label = NodeLabel.CONCEPT.value
    applies_to_rel = RelType.APPLIES_TO.value

    query = f"""
    MATCH (s:{section_label} {{section_id: $section_id}})
    MATCH (c:{concept_label} {{concept_id: $concept_id}})
    MERGE (s)-[r:{applies_to_rel}]->(c)
    SET r.relevance = $relevance
    """

    tx.run(
        query,
        section_id=section_id,
        concept_id=concept_id,
        relevance=relevance,
    )


def create_cites(
    tx: Any,
    source_section_id: str,
    target_section_id: str,
    evidence_text: str,
) -> None:
    """
    Create a CITES relationship between two Section nodes.

    Parameters:
        tx: Active Neo4j transaction object.
        source_section_id: Section that contains the legal reference.
        target_section_id: Section being cited.
        evidence_text: Text evidence that caused the citation edge to be created.

    Returns:
        None
    """
    section_label = NodeLabel.SECTION.value
    cites_rel = RelType.CITES.value

    query = f"""
    MATCH (source:{section_label} {{section_id: $source_section_id}})
    MATCH (target:{section_label} {{section_id: $target_section_id}})
    MERGE (source)-[r:{cites_rel}]->(target)
    SET r.evidence_text = $evidence_text
    """

    tx.run(
        query,
        source_section_id=source_section_id,
        target_section_id=target_section_id,
        evidence_text=evidence_text,
    )


# ---------------------------------------------------------------------------
# Read Operations
# ---------------------------------------------------------------------------

def get_section_by_id(section_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch one Section node by section_id.

    Parameters:
        section_id: Stable Section node ID.

    Returns:
        A dictionary containing section properties, or None if not found.
    """
    driver = get_driver()
    section_label = NodeLabel.SECTION.value

    query = f"""
    MATCH (s:{section_label} {{section_id: $section_id}})
    RETURN s
    LIMIT 1
    """

    with driver.session() as session:
        record = session.run(query, section_id=section_id).single()

    if record is None:
        return None

    return dict(record["s"])


def get_sections_for_concept(concept_name: str) -> List[Dict[str, Any]]:
    """
    Find Section nodes linked to a Concept whose name or aliases match the query.

    This is the main graph entry point for the reasoning agent.

    Parameters:
        concept_name: User-derived concept phrase, such as "fired without notice".

    Returns:
        List of dictionaries containing section data and mapping metadata.
    """
    driver = get_driver()

    section_label = NodeLabel.SECTION.value
    concept_label = NodeLabel.CONCEPT.value
    applies_to_rel = RelType.APPLIES_TO.value

    query = f"""
    MATCH (s:{section_label})-[r:{applies_to_rel}]->(c:{concept_label})
    WHERE
        toLower(c.name) CONTAINS toLower($concept_name)
        OR toLower($concept_name) CONTAINS toLower(c.name)
        OR ANY(alias IN c.aliases WHERE toLower(alias) CONTAINS toLower($concept_name))
        OR ANY(alias IN c.aliases WHERE toLower($concept_name) CONTAINS toLower(alias))
    RETURN
        s,
        c.concept_id AS concept_id,
        c.name AS concept_name,
        r.relevance AS relevance
    ORDER BY
        CASE r.relevance
            WHEN 'primary' THEN 0
            ELSE 1
        END,
        s.act_id,
        s.section_number
    """

    with driver.session() as session:
        records = list(session.run(query, concept_name=concept_name))

    if not records:
        return []

    return [
        {
            "section": dict(record["s"]),
            "concept_id": record["concept_id"],
            "concept_name": record["concept_name"],
            "relevance": record["relevance"],
        }
        for record in records
    ]


def get_neighbors(section_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all nodes directly connected to a Section node.

    This function is used during multi-hop graph expansion.

    Parameters:
        section_id: Stable Section node ID.

    Returns:
        List of dictionaries with:
            node: Neighbor node properties as a dict.
            rel_type: Relationship type as string.
            direction: outgoing or incoming.
    """
    driver = get_driver()
    section_label = NodeLabel.SECTION.value

    query = f"""
    MATCH (s:{section_label} {{section_id: $section_id}})
    CALL {{
        WITH s
        MATCH (s)-[r]->(neighbor)
        RETURN neighbor, type(r) AS rel_type, 'outgoing' AS direction

        UNION

        WITH s
        MATCH (neighbor)-[r]->(s)
        RETURN neighbor, type(r) AS rel_type, 'incoming' AS direction
    }}
    RETURN neighbor, rel_type, direction
    ORDER BY rel_type, direction
    """

    with driver.session() as session:
        records = list(session.run(query, section_id=section_id))

    if not records:
        return []

    return [
        {
            "node": dict(record["neighbor"]),
            "rel_type": record["rel_type"],
            "direction": record["direction"],
        }
        for record in records
    ]


def get_subgraph(section_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all requested Section nodes and all CITES edges between them.

    This is used to assemble the final graph context before LLM synthesis.

    Parameters:
        section_ids: List of Section node IDs to include in the subgraph.

    Returns:
        Dictionary with:
            sections: List of Section node dictionaries.
            cites_edges: List of CITES relationship dictionaries.
    """
    if not section_ids:
        return {
            "sections": [],
            "cites_edges": [],
        }

    driver = get_driver()
    section_label = NodeLabel.SECTION.value
    cites_rel = RelType.CITES.value

    query = f"""
    MATCH (s:{section_label})
    WHERE s.section_id IN $section_ids

    WITH collect(s) AS sections

    OPTIONAL MATCH (source:{section_label})-[r:{cites_rel}]->(target:{section_label})
    WHERE source.section_id IN $section_ids
      AND target.section_id IN $section_ids

    RETURN
        sections,
        collect({{
            source_section_id: source.section_id,
            target_section_id: target.section_id,
            evidence_text: r.evidence_text,
            rel_type: type(r)
        }}) AS cites_edges
    """

    with driver.session() as session:
        record = session.run(query, section_ids=section_ids).single()

    if record is None:
        return {
            "sections": [],
            "cites_edges": [],
        }

    sections = [dict(section) for section in record["sections"]]

    cites_edges = [
        edge
        for edge in record["cites_edges"]
        if edge.get("source_section_id") is not None
        and edge.get("target_section_id") is not None
    ]

    return {
        "sections": sections,
        "cites_edges": cites_edges,
    }


def test_connection() -> None:
    """
    Test Neo4j connectivity with a simple RETURN 1 query.

    Parameters:
        None

    Returns:
        None
    """
    driver = get_driver()

    query = """
    RETURN 1 AS ok
    """

    with driver.session() as session:
        record = session.run(query).single()

    if record and record["ok"] == 1:
        print("Neo4j connection successful.")
    else:
        print("Neo4j connection failed.")


if __name__ == "__main__":
    test_connection()
    print("queries.py loaded successfully.")