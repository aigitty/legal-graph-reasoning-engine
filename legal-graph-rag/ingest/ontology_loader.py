import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = PROJECT_ROOT / "data" / "ontology" / "concept_map.json"


class OntologyLoader:
    """
    Loads concept ontology from concept_map.json into Neo4j.

    Responsibilities:
    1. Create Concept nodes.
    2. Validate mapped Section nodes exist.
    3. Create Section -[:APPLIES_TO]-> Concept relationships.
    """

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def load_concept_map(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read and validate concept_map.json."""
        if not file_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("concept_map.json must contain a list of concept objects.")

        return data

    def create_constraints(self) -> None:
        """Create uniqueness constraint for Concept nodes."""
        query = """
        CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
        FOR (c:Concept)
        REQUIRE c.concept_id IS UNIQUE
        """

        with self.driver.session() as session:
            session.run(query)

    def create_concept_node(self, concept: Dict[str, Any]) -> None:
        """Create or update one Concept node."""
        query = """
        MERGE (c:Concept {concept_id: $concept_id})
        SET
            c.name = $name,
            c.description = $description,
            c.aliases = $aliases
        """

        params = {
            "concept_id": concept["concept_id"],
            "name": concept.get("name", ""),
            "description": concept.get("description", ""),
            "aliases": concept.get("aliases", []),
        }

        with self.driver.session() as session:
            session.run(query, params)

    def section_exists(self, section_id: str) -> bool:
        """Check if a Section node exists before creating relationship."""
        query = """
        MATCH (s:Section {section_id: $section_id})
        RETURN count(s) AS count
        """

        with self.driver.session() as session:
            result = session.run(query, {"section_id": section_id}).single()

        return bool(result and result["count"] > 0)

    def create_applies_to_relationship(
        self,
        section_id: str,
        concept_id: str,
        relevance: str,
    ) -> None:
        """
        Create Section -[:APPLIES_TO]-> Concept relationship.

        Direction:
        Section -> Concept

        Example:
        (:Section {section_id: "IDA_1947_S25F"})
            -[:APPLIES_TO {relevance: "primary"}]->
        (:Concept {concept_id: "wrongful_termination"})
        """
        query = """
        MATCH (s:Section {section_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        MERGE (s)-[r:APPLIES_TO]->(c)
        SET r.relevance = $relevance
        """

        params = {
            "section_id": section_id,
            "concept_id": concept_id,
            "relevance": relevance,
        }

        with self.driver.session() as session:
            session.run(query, params)

    def load(self, concepts: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict[str, str]]]:
        """
        Load all concepts and relationships.

        Returns:
            concepts_loaded: number of Concept nodes processed
            relationships_loaded: number of APPLIES_TO relationships created
            skipped_mappings: section mappings skipped because Section node was missing
        """
        concepts_loaded = 0
        relationships_loaded = 0
        skipped_mappings: List[Dict[str, str]] = []

        for concept in concepts:
            concept_id = concept.get("concept_id")

            if not concept_id:
                raise ValueError(f"Concept missing concept_id: {concept}")

            self.create_concept_node(concept)
            concepts_loaded += 1

            for mapping in concept.get("maps_to", []):
                section_id = mapping.get("section_id")
                relevance = mapping.get("relevance", "supporting")

                if not section_id:
                    skipped_mappings.append({
                        "concept_id": concept_id,
                        "section_id": "MISSING_SECTION_ID",
                        "reason": "Mapping has no section_id",
                    })
                    continue

                if not self.section_exists(section_id):
                    skipped_mappings.append({
                        "concept_id": concept_id,
                        "section_id": section_id,
                        "reason": "Section node not found in Neo4j",
                    })
                    continue

                self.create_applies_to_relationship(
                    section_id=section_id,
                    concept_id=concept_id,
                    relevance=relevance,
                )
                relationships_loaded += 1

        return concepts_loaded, relationships_loaded, skipped_mappings


def main() -> None:
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        raise EnvironmentError("NEO4J_PASSWORD is missing in .env")

    print("=" * 80)
    print("Legal Ontology Loader")
    print("=" * 80)
    print(f"Ontology file: {ONTOLOGY_PATH}")
    print(f"Neo4j URI:     {neo4j_uri}")
    print("=" * 80)

    loader = OntologyLoader(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
    )

    try:
        concepts = loader.load_concept_map(ONTOLOGY_PATH)

        loader.create_constraints()

        concepts_loaded, relationships_loaded, skipped_mappings = loader.load(concepts)

        print("\nOntology loading completed.")
        print("-" * 80)
        print(f"Concept nodes loaded:          {concepts_loaded}")
        print(f"APPLIES_TO relationships made: {relationships_loaded}")
        print(f"Skipped mappings:              {len(skipped_mappings)}")

        if skipped_mappings:
            print("\nSkipped mappings:")
            for item in skipped_mappings:
                print(
                    f"- Concept: {item['concept_id']} | "
                    f"Section: {item['section_id']} | "
                    f"Reason: {item['reason']}"
                )

        print("=" * 80)

    except (Neo4jError, FileNotFoundError, ValueError, EnvironmentError) as error:
        print("\nOntology loading failed.")
        print(f"Error: {error}")
        raise

    finally:
        loader.close()


if __name__ == "__main__":
    main()