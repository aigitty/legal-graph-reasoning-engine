import json
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


BASE_DIR = Path(__file__).resolve().parents[1]

SECTIONS_PATH = (
    BASE_DIR / "data" / "processed" / "sections.jsonl"
)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def load_sections():
    sections = []

    with open(SECTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sections.append(json.loads(line))

    return sections


def create_constraints(tx):
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


def create_act_and_section(tx, section):

    # Create Act node
    tx.run("""
    MERGE (a:Act {act_id: $act_id})

    SET a.act_name = $act_name,
        a.year = $year,
        a.jurisdiction = $jurisdiction,
        a.short_name = $short_name
    """,
    act_id=section["act_id"],
    act_name=section["act_name"],
    year=section["year"],
    jurisdiction=section["jurisdiction"],
    short_name=section["short_name"]
    )

    # Create Section node
    tx.run("""
    MERGE (s:Section {section_id: $section_id})

    SET s.section_number = $section_number,
        s.section_title = $section_title,
        s.section_text = $section_text,
        s.source_file = $source_file
    """,
    section_id=section["section_id"],
    section_number=section["section_number"],
    section_title=section["section_title"],
    section_text=section["section_text"],
    source_file=section["source_file"]
    )

    # Connect Act → Section
    tx.run("""
    MATCH (a:Act {act_id: $act_id})
    MATCH (s:Section {section_id: $section_id})

    MERGE (a)-[:HAS_SECTION]->(s)
    """,
    act_id=section["act_id"],
    section_id=section["section_id"]
    )


def main():

    sections = load_sections()

    print(f"Loaded sections: {len(sections)}")

    with driver.session() as session:

        print("Creating constraints...")
        session.execute_write(create_constraints)

        print("Loading graph into Neo4j...")

        for idx, section in enumerate(sections, start=1):

            session.execute_write(
                create_act_and_section,
                section
            )

            print(
                f"[{idx}/{len(sections)}] "
                f"Loaded {section['section_id']}"
            )

    print("\nGraph loading complete!")

    driver.close()


if __name__ == "__main__":
    main()