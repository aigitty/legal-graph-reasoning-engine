"""
ingest/graph_builder.py

Loads the processed legal graph data into Neo4j.

Current MVP inputs:
  - data/processed/sections.jsonl
      Creates:
        (:Act)
        (:Section)
        (:Act)-[:HAS_SECTION]->(:Section)

  - data/processed/relationships.jsonl
      Creates:
        (:Section)-[:CITES]->(:Section)

This script is intentionally idempotent:
  - Uses MERGE for nodes and relationships
  - Safe to run multiple times
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


# ---------------------------------------------------------------------
# ENV + PATH CONFIG
# ---------------------------------------------------------------------

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SECTIONS_PATH = PROCESSED_DIR / "sections.jsonl"
RELATIONSHIPS_PATH = PROCESSED_DIR / "relationships.jsonl"


# ---------------------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------------------

def validate_env() -> None:
    """Fail early if Neo4j credentials are missing."""
    missing = []

    if not NEO4J_URI:
        missing.append("NEO4J_URI")

    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nPlease check your .env file."
        )


def validate_file_exists(path: Path) -> None:
    """Fail early if required processed file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ---------------------------------------------------------------------
# JSONL LOADERS
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    validate_file_exists(path)

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

    return records


def load_sections() -> list[dict[str, Any]]:
    """Load parsed legal sections from sections.jsonl."""
    return load_jsonl(SECTIONS_PATH)


def load_relationships() -> list[dict[str, Any]]:
    """Load extracted section-to-section relationships from relationships.jsonl."""
    return load_jsonl(RELATIONSHIPS_PATH)


# ---------------------------------------------------------------------
# NEO4J CONSTRAINTS
# ---------------------------------------------------------------------

def create_constraints(tx) -> None:
    """
    Create uniqueness constraints.

    These constraints make MERGE faster and prevent duplicate nodes.
    """
    tx.run("""
    CREATE CONSTRAINT act_id_unique IF NOT EXISTS
    FOR (a:Act)
    REQUIRE a.act_id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT section_id_unique IF NOT EXISTS
    FOR (s:Section)
    REQUIRE s.section_id IS UNIQUE
    """)


# ---------------------------------------------------------------------
# NODE LOADERS
# ---------------------------------------------------------------------

def upsert_act_and_section(tx, section: dict[str, Any]) -> None:
    """
    Create/update one Act node, one Section node,
    and connect them using HAS_SECTION.
    """

    # Create/update Act node
    tx.run(
        """
        MERGE (a:Act {act_id: $act_id})
        SET
            a.act_name = $act_name,
            a.year = $year,
            a.jurisdiction = $jurisdiction,
            a.short_name = $short_name
        """,
        act_id=section["act_id"],
        act_name=section["act_name"],
        year=section["year"],
        jurisdiction=section["jurisdiction"],
        short_name=section["short_name"],
    )

    # Create/update Section node
    tx.run(
        """
        MERGE (s:Section {section_id: $section_id})
        SET
            s.section_number = $section_number,
            s.section_title = $section_title,
            s.section_text = $section_text,
            s.act_id = $act_id,
            s.source_file = $source_file
        """,
        section_id=section["section_id"],
        section_number=section["section_number"],
        section_title=section["section_title"],
        section_text=section["section_text"],
        act_id=section["act_id"],
        source_file=section["source_file"],
    )

    # Create Act -> Section relationship
    tx.run(
        """
        MATCH (a:Act {act_id: $act_id})
        MATCH (s:Section {section_id: $section_id})
        MERGE (a)-[:HAS_SECTION]->(s)
        """,
        act_id=section["act_id"],
        section_id=section["section_id"],
    )


# ---------------------------------------------------------------------
# RELATIONSHIP LOADERS
# ---------------------------------------------------------------------

def upsert_cites_relationship(tx, relationship: dict[str, Any]) -> None:
    """
    Create/update one Section -> Section CITES relationship.

    Expected relationship record:
      {
        "source_section_id": "...",
        "source_act_id": "...",
        "relationship_type": "CITES",
        "target_section_id": "...",
        "target_section_number": "...",
        "evidence_text": "..."
      }
    """

    relationship_type = relationship.get("relationship_type")

    # For MVP, only CITES is produced by relationship_extractor.py.
    # Other relationship types can be added later.
    if relationship_type != "CITES":
        return

    tx.run(
        """
        MATCH (source:Section {section_id: $source_section_id})
        MATCH (target:Section {section_id: $target_section_id})
        MERGE (source)-[r:CITES]->(target)
        SET
            r.evidence_text = $evidence_text,
            r.source_act_id = $source_act_id,
            r.target_section_number = $target_section_number
        """,
        source_section_id=relationship["source_section_id"],
        target_section_id=relationship["target_section_id"],
        source_act_id=relationship.get("source_act_id"),
        target_section_number=relationship.get("target_section_number"),
        evidence_text=relationship.get("evidence_text", ""),
    )


# ---------------------------------------------------------------------
# SUMMARY QUERIES
# ---------------------------------------------------------------------

def print_graph_summary(session) -> None:
    """Print a small summary after loading data."""
    result = session.run(
        """
        MATCH (a:Act)
        OPTIONAL MATCH (a)-[:HAS_SECTION]->(s:Section)
        RETURN
            a.act_id AS act_id,
            a.act_name AS act_name,
            count(s) AS section_count
        ORDER BY a.act_id
        """
    )

    print("\nAct / Section summary:")
    for row in result:
        print(
            f"  {row['act_id']:<10} | "
            f"{row['section_count']:>3} sections | "
            f"{row['act_name']}"
        )

    rel_count = session.run(
        """
        MATCH (:Section)-[r:CITES]->(:Section)
        RETURN count(r) AS cites_count
        """
    ).single()["cites_count"]

    print(f"\nCITES relationships loaded: {rel_count}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    validate_env()
    
    print("\nClearing existing graph data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")

    sections = load_sections()
    relationships = load_relationships()

    print("=" * 80)
    print("Legal Graph Builder")
    print("=" * 80)
    print(f"Sections input:       {SECTIONS_PATH}")
    print(f"Relationships input:  {RELATIONSHIPS_PATH}")
    print(f"Sections found:       {len(sections)}")
    print(f"Relationships found:  {len(relationships)}")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            print("\nCreating constraints...")
            session.execute_write(create_constraints)

            print("\nLoading Act and Section nodes...")
            for index, section in enumerate(sections, start=1):
                session.execute_write(upsert_act_and_section, section)

                if index % 25 == 0 or index == len(sections):
                    print(f"  Loaded sections: {index}/{len(sections)}")

            print("\nLoading CITES relationships...")
            for index, relationship in enumerate(relationships, start=1):
                session.execute_write(upsert_cites_relationship, relationship)

                if index % 25 == 0 or index == len(relationships):
                    print(f"  Loaded relationships: {index}/{len(relationships)}")

            print_graph_summary(session)

        print("\nGraph loading completed successfully.")

    except Neo4jError as error:
        print("\nNeo4j error occurred:")
        print(error)
        raise

    finally:
        driver.close()


if __name__ == "__main__":
    main()
