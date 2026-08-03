"""
graph/act_registry.py

Offline reader for data/ontology/act_metadata.json.

The same temporal/territorial facts are stored on the Act nodes in Neo4j (written
by ingest/act_metadata_loader.py) and joined onto every section by
graph/queries.py. This module exists for the things that need those facts
WITHOUT a database round-trip:

  - resolving a free-text place name to a jurisdiction ("bangalore" -> "Karnataka")
  - explaining a suppression to the user ("repealed by section 69 of the Code on
    Wages, 2019") without re-querying per section
  - letting tests and tools/build_concept_map.py reason about in-force status
    with no Neo4j available

No LLM, no network, no Neo4j. Read-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ACT_METADATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ontology" / "act_metadata.json"
)

CENTRAL = "Central"


@dataclass(frozen=True)
class ActMeta:
    act_id: str
    act_name: str
    short_name: str
    jurisdiction: str
    status: str  # "in_force" | "repealed"
    in_force_from: str
    repeals: tuple[str, ...]
    repealed_by: str
    repeal_authority: str
    act_priority: int

    @property
    def is_in_force(self) -> bool:
        return self.status == "in_force"

    @property
    def is_central(self) -> bool:
        return self.jurisdiction == CENTRAL


def _load() -> tuple[dict[str, ActMeta], dict[str, str]]:
    with ACT_METADATA_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    acts: dict[str, ActMeta] = {}
    for entry in raw["acts"]:
        acts[entry["act_id"]] = ActMeta(
            act_id=entry["act_id"],
            act_name=entry.get("act_name", ""),
            short_name=entry.get("short_name", ""),
            jurisdiction=entry.get("jurisdiction", CENTRAL),
            status=entry.get("status", "in_force"),
            in_force_from=entry.get("in_force_from") or "",
            repeals=tuple(entry.get("repeals") or ()),
            repealed_by=entry.get("repealed_by") or "",
            repeal_authority=entry.get("repeal_authority") or "",
            act_priority=int(entry.get("act_priority") or 0),
        )

    aliases = {
        str(key).lower(): value
        for key, value in (raw.get("jurisdiction_aliases") or {}).items()
    }
    return acts, aliases


ACTS, _JURISDICTION_ALIASES = _load()

REPEALED_ACT_IDS: frozenset[str] = frozenset(
    act_id for act_id, meta in ACTS.items() if not meta.is_in_force
)

MAX_ACT_PRIORITY: int = max((meta.act_priority for meta in ACTS.values()), default=1) or 1


def get_act(act_id: str) -> Optional[ActMeta]:
    """Return metadata for an act_id, or None if the act is not in the corpus."""
    return ACTS.get(act_id)


def act_id_for_section(section_id: str) -> str:
    """
    Derive the act_id from a section_id ("COW_2019_S18" -> "COW_2019").

    Section ids are built as <ACT_ID>_S<NUMBER> by ingest/graph_builder.py, so
    this is a pure string operation — no lookup needed. Returns "" if the id does
    not have that shape.
    """
    if "_S" not in section_id:
        return ""
    return section_id.rsplit("_S", 1)[0]


def normalize_jurisdiction(value: str | None) -> Optional[str]:
    """
    Resolve a free-text place name to a jurisdiction used by the Act nodes.

    "karnataka" / "Bangalore" / "bengaluru" -> "Karnataka"
    "india" / "central"                     -> "Central"
    "Maharashtra"                           -> "Maharashtra"  (title-cased as-is;
                                               no Act in the corpus matches it,
                                               which is the correct outcome —
                                               Karnataka law is then filtered out)
    None / ""                               -> None (jurisdiction not stated)

    Returning the unknown state unchanged rather than falling back to "Central"
    is deliberate: it lets the caller distinguish "user named a state we have no
    law for" from "user named no state at all", which get different treatment in
    agents/nodes/retrieval_node.py.
    """
    if not value:
        return None

    text = " ".join(str(value).strip().lower().split())
    if not text:
        return None

    if text in _JURISDICTION_ALIASES:
        return _JURISDICTION_ALIASES[text]

    # Allow a state name embedded in a longer phrase ("I work in Bangalore").
    for alias, jurisdiction in _JURISDICTION_ALIASES.items():
        if alias in text:
            return jurisdiction

    return text.title()


_SECTION_REF_RE = re.compile(
    r"\b(?:section|sections|sec\.?|s\.)\s*([0-9]{1,3}[A-Za-z]{0,4})\b",
    re.IGNORECASE,
)


def resolve_section_references(text: str) -> list[str]:
    """
    Extract explicit section references from a query and resolve them to
    candidate section_ids.

    "what does Section 25N of the Industrial Disputes Act say?"
        -> ["IDA_1947_S25N"]
    "section 18"  (no act named)
        -> ["COW_2019_S18", "IDA_1947_S18", ...]  one candidate per in-force act

    A user naming a section is asking a LOOKUP question, not a situational one.
    Concept grounding cannot serve it: "Section 25N" grounds to `industrial
    dispute`, which retrieves the dispute-machinery sections and never returns
    25N itself. The caller checks these candidates against the graph and keeps
    the ones that exist, so an unresolvable reference simply yields nothing
    rather than a wrong section.

    Repealed acts are excluded — a reference to a repealed provision must not
    resurrect it. Returns [] when the query names no section.
    """
    if not text:
        return []

    numbers = [match.group(1).upper() for match in _SECTION_REF_RE.finditer(text)]
    if not numbers:
        return []

    lowered = text.lower()

    # Which acts did the user name? Match on full name, short name, and the
    # distinctive word in each title so "Industrial Disputes Act", "IDA" and
    # "the Gratuity Act" all resolve.
    named: list[str] = []
    for act_id, meta in ACTS.items():
        if not meta.is_in_force:
            continue
        needles = {
            meta.act_name.lower(),
            meta.short_name.lower(),
            meta.act_name.lower().replace(",", ""),
        }
        # Drop the trailing year and generic words to get a usable keyword.
        core = (
            meta.act_name.lower()
            .replace("act", "")
            .replace("code on", "code on")
            .replace(",", "")
            .strip()
        )
        if core:
            needles.add(core)
        if any(needle and needle in lowered for needle in needles):
            named.append(act_id)

    target_acts = named or [
        act_id for act_id, meta in ACTS.items() if meta.is_in_force
    ]

    candidates: list[str] = []
    for number in numbers:
        for act_id in target_acts:
            section_id = f"{act_id}_S{number}"
            if section_id not in candidates:
                candidates.append(section_id)
    return candidates


def is_section_in_force(section_id: str) -> bool:
    """True unless the section's Act is marked repealed."""
    return act_id_for_section(section_id) not in REPEALED_ACT_IDS


def in_force_acts() -> list[ActMeta]:
    """Acts currently in force, highest priority first, then by name."""
    return sorted(
        (meta for meta in ACTS.values() if meta.is_in_force),
        key=lambda meta: (-meta.act_priority, meta.act_name),
    )


def scope_sentence() -> str:
    """
    One sentence naming the law this engine can actually answer from.

    DERIVED, never hardcoded. The user-facing "here is my scope" text used to be
    a literal string in two different nodes, both of which listed the Minimum
    Wages Act 1948 — so the moment that Act was marked repealed, the engine was
    simultaneously refusing to cite it and advertising it as in scope. Building
    the sentence from the same metadata that drives the filtering keeps the two
    honest with each other.
    """
    names = [meta.act_name for meta in in_force_acts()]
    if not names:
        return "This engine has no statutes loaded."
    if len(names) == 1:
        listed = names[0]
    else:
        listed = ", ".join(names[:-1]) + " and the " + names[-1]
    return (
        "This engine answers questions about Indian employment and labour law "
        f"as covered by the {listed}."
    )


def repeal_note(act_id: str) -> str:
    """
    One human-readable sentence explaining why an Act was suppressed, or "" if it
    is still in force. Used to tell the user what happened rather than silently
    dropping law.
    """
    meta = ACTS.get(act_id)
    if meta is None or meta.is_in_force:
        return ""

    repealing = ACTS.get(meta.repealed_by)
    repealing_name = repealing.act_name if repealing else meta.repealed_by
    authority = meta.repeal_authority

    note = f"The {meta.act_name} has been repealed by the {repealing_name}"
    if authority:
        number = authority.rsplit("_S", 1)[-1] if "_S" in authority else ""
        if number:
            note += f" (section {number})"
    return note + "."


if __name__ == "__main__":
    print("Acts in the corpus:")
    for act_id, meta in sorted(ACTS.items()):
        flag = "in force" if meta.is_in_force else f"REPEALED by {meta.repealed_by}"
        print(f"  {act_id:<12} {meta.jurisdiction:<10} prio={meta.act_priority}  {flag}")

    print("\nJurisdiction resolution:")
    for probe in ["Karnataka", "bangalore", "I work in Bengaluru", "Maharashtra",
                  "India", None, ""]:
        print(f"  {probe!r:<26} -> {normalize_jurisdiction(probe)!r}")

    print("\nRepeal notes:")
    for act_id in sorted(REPEALED_ACT_IDS):
        print(f"  {repeal_note(act_id)}")
