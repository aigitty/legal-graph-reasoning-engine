"""
ingest/act_metadata_loader.py

Loads data/ontology/act_metadata.json onto the Act nodes already created by
graph_builder.py, and materialises the OVERRIDES edges the schema has always
declared but never had data for.

This is an ADDITIVE ingest step. It does not touch Section or Concept nodes and
it never creates an Act — it only annotates Acts that graph_builder.py already
loaded, so running it against an unbuilt graph is a no-op with a warning.

What it writes onto each (:Act):
    jurisdiction      "Central" | a state name — territorial reach
    status            "in_force" | "repealed"
    in_force_from     ISO date string
    repealed_by       act_id of the repealing Act, or null
    repeal_authority  section_id of the repealing provision, or null
    act_priority      int, higher = preferred when two Acts both apply

And between Acts:
    (:Act {repealing})-[:OVERRIDES {authority}]->(:Act {repealed})

Why this exists
---------------
Before this step the engine had no way to know that the Code on Wages 2019
repealed the Minimum Wages Act 1948, so it cited both side by side as though
both were simultaneously operative. It also could not know that the Karnataka
Shops Act is state legislation, so it cited Karnataka law to users in other
states. Both facts now live in the graph and are read at query time by
graph/queries.py and enforced in agents/nodes/retrieval_node.py.

Run:
    python -m ingest.act_metadata_loader
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACT_METADATA_PATH = PROJECT_ROOT / "data" / "ontology" / "act_metadata.json"


class ActMetadataLoader:
    """Annotates Act nodes with temporal/territorial metadata + OVERRIDES edges."""

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def load_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read and validate act_metadata.json, returning the acts list."""
        if not file_path.exists():
            raise FileNotFoundError(f"Act metadata file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        acts = data.get("acts")
        if not isinstance(acts, list) or not acts:
            raise ValueError("act_metadata.json must contain a non-empty 'acts' list.")

        for act in acts:
            if not act.get("act_id"):
                raise ValueError(f"Act entry missing act_id: {act}")
            status = act.get("status")
            if status not in {"in_force", "repealed"}:
                raise ValueError(
                    f"Act {act['act_id']} has invalid status {status!r} "
                    f"(expected 'in_force' or 'repealed')."
                )
            # A repealed Act must name both what repealed it and under which
            # provision — that section_id is checked against the graph below.
            if status == "repealed" and not act.get("repeal_authority"):
                raise ValueError(
                    f"Act {act['act_id']} is marked repealed but has no "
                    f"repeal_authority section_id."
                )

        return acts

    def act_exists(self, act_id: str) -> bool:
        query = "MATCH (a:Act {act_id: $act_id}) RETURN count(a) AS count"
        with self.driver.session() as session:
            record = session.run(query, {"act_id": act_id}).single()
        return bool(record and record["count"] > 0)

    def section_exists(self, section_id: str) -> bool:
        query = "MATCH (s:Section {section_id: $section_id}) RETURN count(s) AS count"
        with self.driver.session() as session:
            record = session.run(query, {"section_id": section_id}).single()
        return bool(record and record["count"] > 0)

    def annotate_act(self, act: Dict[str, Any]) -> None:
        """Write the temporal/territorial properties onto one Act node."""
        query = """
        MATCH (a:Act {act_id: $act_id})
        SET
            a.jurisdiction     = $jurisdiction,
            a.status           = $status,
            a.in_force_from    = $in_force_from,
            a.repealed_by      = $repealed_by,
            a.repeal_authority = $repeal_authority,
            a.act_priority     = $act_priority
        """

        params = {
            "act_id": act["act_id"],
            "jurisdiction": act.get("jurisdiction", "Central"),
            "status": act.get("status", "in_force"),
            "in_force_from": act.get("in_force_from"),
            "repealed_by": act.get("repealed_by"),
            "repeal_authority": act.get("repeal_authority"),
            "act_priority": int(act.get("act_priority", 0)),
        }

        with self.driver.session() as session:
            session.run(query, params)

    def create_overrides(
        self, repealing_act_id: str, repealed_act_id: str, authority: str | None
    ) -> None:
        """
        Create (:Act {repealing})-[:OVERRIDES]->(:Act {repealed}).

        Direction follows graph/schema.py: the NEWER act points at the one it
        supersedes.
        """
        query = """
        MATCH (newer:Act {act_id: $repealing_act_id})
        MATCH (older:Act {act_id: $repealed_act_id})
        MERGE (newer)-[r:OVERRIDES]->(older)
        SET r.authority = $authority
        """

        with self.driver.session() as session:
            session.run(
                query,
                {
                    "repealing_act_id": repealing_act_id,
                    "repealed_act_id": repealed_act_id,
                    "authority": authority,
                },
            )

    def load(
        self, acts: List[Dict[str, Any]]
    ) -> Tuple[int, int, List[Dict[str, str]]]:
        """
        Annotate every Act and build the OVERRIDES edges.

        Returns:
            acts_annotated:      Act nodes updated
            overrides_created:   OVERRIDES edges written
            skipped:             entries skipped, with the reason
        """
        acts_annotated = 0
        overrides_created = 0
        skipped: List[Dict[str, str]] = []

        for act in acts:
            act_id = act["act_id"]

            if not self.act_exists(act_id):
                skipped.append(
                    {
                        "act_id": act_id,
                        "target": "-",
                        "reason": "Act node not found in Neo4j (not ingested)",
                    }
                )
                continue

            # A repeal claim must be backed by a provision that really exists in
            # the graph, otherwise we would be suppressing law on the strength of
            # an unverifiable assertion.
            authority = act.get("repeal_authority")
            if act.get("status") == "repealed" and not self.section_exists(authority):
                skipped.append(
                    {
                        "act_id": act_id,
                        "target": str(authority),
                        "reason": "repeal_authority section not found in Neo4j — "
                        "refusing to mark the Act repealed",
                    }
                )
                continue

            self.annotate_act(act)
            acts_annotated += 1

            for repealed_act_id in act.get("repeals", []) or []:
                if not self.act_exists(repealed_act_id):
                    # Expected for acts the corpus does not contain (e.g. the
                    # Payment of Bonus Act 1965). Not an error.
                    skipped.append(
                        {
                            "act_id": act_id,
                            "target": repealed_act_id,
                            "reason": "OVERRIDES target not in corpus (no Act node)",
                        }
                    )
                    continue

                self.create_overrides(act_id, repealed_act_id, authority)
                overrides_created += 1

        return acts_annotated, overrides_created, skipped


def main() -> None:
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        raise EnvironmentError("NEO4J_PASSWORD is missing in .env")

    print("=" * 80)
    print("Act Metadata Loader — temporal + territorial annotations")
    print("=" * 80)
    print(f"Metadata file: {ACT_METADATA_PATH}")
    print(f"Neo4j URI:     {neo4j_uri}")
    print("=" * 80)

    loader = ActMetadataLoader(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
    )

    try:
        acts = loader.load_metadata(ACT_METADATA_PATH)
        acts_annotated, overrides_created, skipped = loader.load(acts)

        print("\nAct metadata loading completed.")
        print("-" * 80)
        print(f"Act nodes annotated:      {acts_annotated}")
        print(f"OVERRIDES edges created:  {overrides_created}")
        print(f"Skipped:                  {len(skipped)}")

        if skipped:
            print("\nSkipped entries:")
            for item in skipped:
                print(
                    f"- Act: {item['act_id']} | Target: {item['target']} | "
                    f"Reason: {item['reason']}"
                )

        print("=" * 80)

    except (Neo4jError, FileNotFoundError, ValueError, EnvironmentError) as error:
        print("\nAct metadata loading failed.")
        print(f"Error: {error}")
        raise

    finally:
        loader.close()


if __name__ == "__main__":
    main()
