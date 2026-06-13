"""
agents/ontology.py

Loads data/ontology/concept_map.json and provides deterministic concept
grounding: mapping free-text queries to canonical Concept.name values that
exist as Concept nodes in Neo4j (loaded by ingest/ontology_loader.py).

No LLM. Fuzzy fallback uses only the Python standard library (difflib),
so this module introduces zero new dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

CONCEPT_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ontology" / "concept_map.json"
)

FUZZY_CUTOFF = 0.8


@dataclass(frozen=True)
class ConceptEntry:
    concept_id: str
    name: str
    aliases: tuple[str, ...]
    section_relevance: dict[str, str]  # section_id -> "primary" | "supporting"

    @property
    def match_terms(self) -> tuple[str, ...]:
        return (self.name.lower(),) + tuple(a.lower() for a in self.aliases)


def _load_concepts() -> list[ConceptEntry]:
    with open(CONCEPT_MAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    concepts: list[ConceptEntry] = []
    for entry in raw:
        relevance = {
            m["section_id"]: m.get("relevance", "supporting")
            for m in entry.get("maps_to", [])
        }
        concepts.append(
            ConceptEntry(
                concept_id=entry["concept_id"],
                name=entry["name"],
                aliases=tuple(entry.get("aliases", [])),
                section_relevance=relevance,
            )
        )
    return concepts


CONCEPTS: list[ConceptEntry] = _load_concepts()

# Flat lookup for fuzzy fallback: lowercased term -> canonical concept name
_ALL_TERMS: dict[str, str] = {
    term: concept.name for concept in CONCEPTS for term in concept.match_terms
}


def ground_query(text: str) -> list[str]:
    """
    Return canonical Concept.name values matched in `text`.

    Strategy:
      1. Substring match: any concept name/alias found inside the
         lowercased, whitespace-normalized query grounds that concept.
         Multiple concepts can match one query.
      2. If nothing matches, fall back to a single closest fuzzy match
         (difflib) against all names/aliases, above FUZZY_CUTOFF.

    Returns an empty list if nothing matches either way.
    """
    query = " ".join(text.lower().split())
    if not query:
        return []

    matched: list[str] = []
    seen: set[str] = set()

    for concept in CONCEPTS:
        for term in concept.match_terms:
            if term and term in query:
                if concept.name not in seen:
                    seen.add(concept.name)
                    matched.append(concept.name)
                break

    if matched:
        return matched

    close = get_close_matches(query, _ALL_TERMS.keys(), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return [_ALL_TERMS[close[0]]]

    return []


def relevance_for(concept_names: list[str]) -> dict[str, str]:
    """
    Merge section relevance across the given concept names.

    "primary" wins over "supporting" if a section appears under multiple
    concepts with different relevance.
    """
    merged: dict[str, str] = {}
    for concept in CONCEPTS:
        if concept.name not in concept_names:
            continue
        for section_id, relevance in concept.section_relevance.items():
            if merged.get(section_id) != "primary":
                merged[section_id] = relevance
    return merged


if __name__ == "__main__":
    # Quick offline check — no Neo4j/network needed.
    test_queries = [
        "gratuity",
        "I was fired without notice after 3 years in Karnataka",
        "my employer hasn't paid me for two months",
        "what is minimum wage for my job",
        "asdkjasjdas nonsense query",
    ]
    for q in test_queries:
        print(f"{q!r:60} -> {ground_query(q)}")