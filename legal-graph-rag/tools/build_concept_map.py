"""
tools/build_concept_map.py

Rebuilds data/ontology/concept_map.json from three inputs:

  1. the EXISTING concept_map.json          — descriptions, aliases and the
                                              mappings already curated there
  2. tools/new_concepts.json                — additional concepts to define
  3. tools/section_concept_map.json         — section -> [(concept, relevance)]

and validates the result against the real corpus before writing.

Why a builder instead of hand-editing the JSON
----------------------------------------------
The ontology is the ONLY entry point into the graph: a section that no concept
maps to is unreachable by any query. Before this, 59 of 255 sections were
reachable. Getting from there to full coverage means touching a few hundred
mappings, which is exactly the kind of edit that silently rots a hand-maintained
JSON. Keeping the source of truth in a flat section->concept table means the
review question is "is this section tagged right?" and the invariants below are
machine-checked on every rebuild.

Invariants enforced (build fails if any is violated)
----------------------------------------------------
  * every mapped section_id exists in data/processed/sections.jsonl
  * every concept_id referenced by a mapping is actually defined
  * relevance is 'primary' or 'supporting'
  * every concept has at least one IN-FORCE primary anchor
    -> this is the one that matters most. Traversal only expands from primary
       anchors and only scores confidence 1.0 when a primary exists, so a
       concept whose primaries all sit in a repealed Act (e.g. minimum_wages
       anchored on MWA_1948 after COW_2019 repealed it) would silently degrade
       to a weak, unexpandable retrieval. See A.3 / retrieval_node filtering.

Run:
    python -m tools.build_concept_map            # validate + write
    python -m tools.build_concept_map --dry-run  # validate + report only
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONCEPT_MAP_PATH = PROJECT_ROOT / "data" / "ontology" / "concept_map.json"
ACT_METADATA_PATH = PROJECT_ROOT / "data" / "ontology" / "act_metadata.json"
SECTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "sections.jsonl"
NEW_CONCEPTS_PATH = Path(__file__).resolve().parent / "new_concepts.json"
SECTION_MAP_PATH = Path(__file__).resolve().parent / "section_concept_map.json"

VALID_RELEVANCE = {"primary", "supporting"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_sections() -> Dict[str, Dict[str, Any]]:
    sections: Dict[str, Dict[str, Any]] = {}
    with SECTIONS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                sections[record["section_id"]] = record
    return sections


def _repealed_act_ids() -> set[str]:
    metadata = _load_json(ACT_METADATA_PATH)
    return {
        act["act_id"]
        for act in metadata["acts"]
        if act.get("status") == "repealed"
    }


def _act_jurisdictions() -> Dict[str, str]:
    metadata = _load_json(ACT_METADATA_PATH)
    return {
        act["act_id"]: act.get("jurisdiction", "Central")
        for act in metadata["acts"]
    }


def build() -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Returns (concepts, errors, stats). `concepts` is the merged ontology ready to
    write; `errors` is empty on a clean build.
    """
    existing: List[Dict[str, Any]] = _load_json(CONCEPT_MAP_PATH)
    new_concepts: List[Dict[str, Any]] = _load_json(NEW_CONCEPTS_PATH)
    section_map: Dict[str, List[List[str]]] = _load_json(SECTION_MAP_PATH)["mappings"]

    sections = _load_sections()
    repealed_acts = _repealed_act_ids()
    act_jurisdiction = _act_jurisdictions()

    # ---- assemble the concept list (existing first, order preserved) --------
    concepts: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    for concept in existing:
        entry = {
            "concept_id": concept["concept_id"],
            "name": concept["name"],
            "description": concept.get("description", ""),
            "aliases": list(concept.get("aliases", [])),
            "maps_to": list(concept.get("maps_to", [])),
        }
        concepts.append(entry)
        by_id[entry["concept_id"]] = entry

    for concept in new_concepts:
        if concept["concept_id"] in by_id:
            continue
        entry = {
            "concept_id": concept["concept_id"],
            "name": concept["name"],
            "description": concept.get("description", ""),
            "aliases": list(concept.get("aliases", [])),
            "maps_to": [],
        }
        concepts.append(entry)
        by_id[entry["concept_id"]] = entry

    # ---- merge extra aliases onto existing concepts -------------------------
    errors: List[str] = []
    aliases_added = 0

    extra_aliases = _load_json(SECTION_MAP_PATH).get("extra_aliases", {})
    for concept_id, aliases in extra_aliases.items():
        if concept_id.startswith("_"):
            continue
        if concept_id not in by_id:
            errors.append(f"extra_aliases: unknown concept_id {concept_id!r}")
            continue
        existing_aliases = {a.lower() for a in by_id[concept_id]["aliases"]}
        for alias in aliases:
            if alias.lower() not in existing_aliases:
                by_id[concept_id]["aliases"].append(alias)
                existing_aliases.add(alias.lower())
                aliases_added += 1

    # ---- merge the section -> concept table --------------------------------
    added = 0
    upgraded = 0

    for section_id, pairs in section_map.items():
        # Underscore-prefixed keys are section-block comments, not mappings.
        if section_id.startswith("_"):
            continue
        if section_id not in sections:
            errors.append(f"{section_id}: not present in sections.jsonl")
            continue

        for pair in pairs:
            concept_id, relevance = pair[0], pair[1]

            if concept_id not in by_id:
                errors.append(f"{section_id}: unknown concept_id {concept_id!r}")
                continue
            if relevance not in VALID_RELEVANCE:
                errors.append(
                    f"{section_id} -> {concept_id}: bad relevance {relevance!r}"
                )
                continue

            maps_to = by_id[concept_id]["maps_to"]
            current = next(
                (m for m in maps_to if m["section_id"] == section_id), None
            )
            if current is None:
                maps_to.append({"section_id": section_id, "relevance": relevance})
                added += 1
            elif current["relevance"] != "primary" and relevance == "primary":
                # 'primary' always wins over an existing 'supporting'.
                current["relevance"] = "primary"
                upgraded += 1

    # ---- sort each concept's mappings: primaries first, then by id ---------
    for concept in concepts:
        concept["maps_to"].sort(
            key=lambda m: (0 if m["relevance"] == "primary" else 1, m["section_id"])
        )

    # ---- invariant: every concept needs a live, reachable primary anchor ----
    #
    # Two filters run over the retrieval set at query time — repealed Acts are
    # dropped for everyone, and state Acts are dropped for users outside that
    # state. A concept whose primaries survive neither filter is dead law for
    # the user asking. So the anchor must be in force AND (unless the concept is
    # declared state-only) available to a Central/unspecified-jurisdiction user.
    state_only = set(_load_json(SECTION_MAP_PATH).get("state_only_concepts", []))

    for concept in concepts:
        primaries = [m for m in concept["maps_to"] if m["relevance"] == "primary"]
        in_force = [
            m for m in primaries
            if sections[m["section_id"]]["act_id"] not in repealed_acts
        ]
        if not in_force:
            errors.append(
                f"concept {concept['concept_id']!r} has no IN-FORCE primary anchor "
                f"(primaries: {[m['section_id'] for m in primaries] or 'none'}). "
                f"Traversal cannot expand from it and confidence never reaches 1.0."
            )
            continue

        if concept["concept_id"] in state_only:
            continue

        central = [
            m for m in in_force
            if act_jurisdiction.get(sections[m["section_id"]]["act_id"]) == "Central"
        ]
        if not central:
            errors.append(
                f"concept {concept['concept_id']!r} has no in-force CENTRAL primary "
                f"anchor — its primaries "
                f"({[m['section_id'] for m in in_force]}) are all state legislation, "
                f"so jurisdiction filtering leaves a user outside that state with no "
                f"anchor. Either map a Central provision as primary, or add the "
                f"concept to 'state_only_concepts' in section_concept_map.json."
            )

    # ---- coverage stats ----------------------------------------------------
    mapped = {m["section_id"] for c in concepts for m in c["maps_to"]}
    live_sections = {
        sid for sid, s in sections.items() if s["act_id"] not in repealed_acts
    }
    live_mapped = mapped & live_sections

    per_act: Dict[str, Dict[str, int]] = defaultdict(lambda: {"mapped": 0, "total": 0})
    for sid, section in sections.items():
        per_act[section["act_id"]]["total"] += 1
        if sid in mapped:
            per_act[section["act_id"]]["mapped"] += 1

    stats = {
        "concepts": len(concepts),
        "mappings": sum(len(c["maps_to"]) for c in concepts),
        "added": added,
        "upgraded": upgraded,
        "aliases_added": aliases_added,
        "sections_total": len(sections),
        "sections_mapped": len(mapped),
        "live_sections_total": len(live_sections),
        "live_sections_mapped": len(live_mapped),
        "unmapped_live": sorted(live_sections - mapped),
        "per_act": dict(per_act),
    }

    return concepts, errors, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report without writing concept_map.json",
    )
    args = parser.parse_args()

    concepts, errors, stats = build()

    print("=" * 78)
    print("Concept map builder")
    print("=" * 78)
    print(f"Concepts:              {stats['concepts']}")
    print(f"APPLIES_TO mappings:   {stats['mappings']}  "
          f"(+{stats['added']} new, {stats['upgraded']} upgraded to primary)")
    print(f"Aliases added:         {stats['aliases_added']}")
    print(
        f"Section coverage:      {stats['sections_mapped']}/{stats['sections_total']} "
        f"all sections"
    )
    print(
        f"                       {stats['live_sections_mapped']}/"
        f"{stats['live_sections_total']} IN-FORCE sections "
        f"({100 * stats['live_sections_mapped'] / stats['live_sections_total']:.0f}%)"
    )
    print("\nPer act:")
    for act_id, counts in sorted(stats["per_act"].items()):
        print(f"  {act_id:<12} {counts['mapped']:>3}/{counts['total']:<3}")

    if stats["unmapped_live"]:
        print(f"\nUnmapped in-force sections ({len(stats['unmapped_live'])}):")
        print("  " + ", ".join(stats["unmapped_live"]))

    if errors:
        print(f"\nBUILD FAILED — {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    if args.dry_run:
        print("\n[dry run] concept_map.json NOT written.")
        return

    with CONCEPT_MAP_PATH.open("w", encoding="utf-8") as handle:
        json.dump(concepts, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"\nWrote {CONCEPT_MAP_PATH}")
    print("Next: python -m ingest.ontology_loader   (pushes it into Neo4j)")


if __name__ == "__main__":
    main()
