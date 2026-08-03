"""
tests/verify.py

End-to-end verification harness for the Legal Graph RAG pipeline.

Runs every query in tests/golden_queries.json through the real pipeline (real
Neo4j, real Gemini) and checks the result against the RAW CORPUS —
data/processed/sections.jsonl, data/ontology/concept_map.json and
data/ontology/act_metadata.json. Nothing is compared against a previous run's
output, so a regression cannot be "blessed" by re-recording it.

Two classes of check run on every query:

  UNIVERSAL INVARIANTS (applied to all queries, no per-query config needed)
    * every verified citation is a real Section in sections.jsonl
    * every verified citation was in the retrieval set shown to the model
      (provenance — the model may only cite what it was given)
    * no citation belongs to a REPEALED Act
    * no citation belongs to a STATE Act outside the query's jurisdiction
    * no unverified [SECTION_ID] marker survives into the visible answer
    * terminal exits (out_of_domain / rejected / error) carry no citations
    * ok / insufficient_evidence answers carry the disclaimer

  PER-QUERY EXPECTATIONS from golden_queries.json (status, must_cite,
  must_retrieve, forbid_cite, forbid_acts, expect_concepts, expect_text,
  forbid_text, min_confidence).

Usage:
    python -m tests.verify                 # run everything
    python -m tests.verify --only T04 T09  # run specific ids
    python -m tests.verify --workers 4     # parallelism (default 4)
    python -m tests.verify --verbose       # print every final answer
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from agents.citations import CITATION_RE, looks_like_citation  # noqa: E402
from agents.persona import CITIZEN, normalize_persona  # noqa: E402
from graph import act_registry  # noqa: E402
from graph_agent import run  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_queries.json"
SECTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "sections.jsonl"

TERMINAL_NO_CONTENT = {"out_of_domain", "rejected", "error"}


def _markers(text: str) -> list[str]:
    """
    Base section ids of every citation marker in `text`.

    Uses the SAME pattern as the pipeline (agents/citations.py) so the harness
    detects exactly what the stripper could have missed — including sub-section
    forms like [IDA_1947_S7(1)], which an earlier, narrower pattern in both the
    pipeline and this harness silently ignored, letting unverified markers reach
    the user while every check reported green.
    """
    return [
        match.group(1)
        for match in CITATION_RE.finditer(text)
        if looks_like_citation(match.group(1))
    ]


def load_corpus() -> dict[str, dict[str, Any]]:
    """All real sections, keyed by section_id. This is the ground truth."""
    corpus: dict[str, dict[str, Any]] = {}
    with SECTIONS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                corpus[record["section_id"]] = record
    return corpus


CORPUS = load_corpus()


@dataclass
class Result:
    case_id: str
    query: str
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    status: str = ""
    confidence: float = 0.0
    grounded: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    answer: str = ""
    elapsed: float = 0.0
    error: str = ""
    numeric_notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)


# ---------------------------------------------------------------------------
# Universal invariants — these hold for EVERY query regardless of expectations
# ---------------------------------------------------------------------------

def check_invariants(result: Result, state: Any, case: dict[str, Any]) -> None:
    verified = list(state.verified_section_ids or [])
    retrieved = set(state.retrieval.section_ids) if state.retrieval else set()
    answer = state.final_answer or ""

    # 1. Every cited section must be a real section in the corpus.
    for section_id in verified:
        if section_id not in CORPUS:
            result.fail(
                f"HALLUCINATED CITATION {section_id!r} — not in sections.jsonl"
            )

    # 2. Provenance: the model may only cite what it was shown.
    for section_id in verified:
        if section_id not in retrieved:
            result.fail(
                f"PROVENANCE VIOLATION {section_id!r} — cited but not in the "
                f"retrieval set"
            )

    # 3. No repealed law may be cited.
    for section_id in verified:
        act_id = act_registry.act_id_for_section(section_id)
        if act_id in act_registry.REPEALED_ACT_IDS:
            result.fail(
                f"REPEALED LAW CITED {section_id!r} — {act_registry.repeal_note(act_id)}"
            )

    # 4. No out-of-jurisdiction state law may be cited.
    hint = act_registry.normalize_jurisdiction(case.get("jurisdiction_hint"))
    if hint:
        for section_id in verified:
            act_id = act_registry.act_id_for_section(section_id)
            meta = act_registry.get_act(act_id)
            if meta and not meta.is_central and meta.jurisdiction != hint:
                result.fail(
                    f"WRONG JURISDICTION {section_id!r} — {meta.act_name} applies "
                    f"only in {meta.jurisdiction}, query concerns {hint}"
                )

    # 5. No unverified marker may survive into the visible answer.
    verified_set = set(verified)
    for token in _markers(answer):
        if token not in verified_set:
            result.fail(
                f"UNVERIFIED MARKER {token!r} leaked into the final answer"
            )

    # 6. Citizen answers strip ALL markers; lawyer answers keep verified ones.
    persona = normalize_persona(case.get("persona"))
    if persona == CITIZEN:
        leaked = _markers(answer)
        if leaked:
            result.fail(
                f"CITIZEN ANSWER shows raw section ids {leaked} — should be "
                f"rendered in words"
            )

    # 7. Safety / domain exits must carry no legal content.
    if state.status in TERMINAL_NO_CONTENT and verified:
        result.fail(
            f"status={state.status} but {len(verified)} citation(s) reached the "
            f"user: {verified}"
        )

    # 8. Substantive answers must carry the disclaimer.
    if state.status in {"ok", "insufficient_evidence"}:
        if "not legal advice" not in answer.lower():
            result.fail("Disclaimer missing from a substantive answer")

    # 9. An answer must actually exist.
    if not answer.strip():
        result.fail("final_answer is empty")


# ---------------------------------------------------------------------------
# Numeric claim cross-check
# ---------------------------------------------------------------------------
#
# The invariants above prove every CITATION is real. They say nothing about
# whether the PROSE is faithful to it — and in a legal answer the numbers are
# what a user acts on: thresholds ("one year of continuous service"), periods
# ("within three years"), multipliers ("twice the ordinary rate"), caps ("ten
# times the claim"). A fabricated number is the highest-impact error the system
# can make while still passing every citation check.
#
# So: pull every quantity out of the answer and require it to appear either in
# the text of a section that was actually cited, or in the user's own query.
# Reported as ADVISORY rather than a hard failure, because legitimate prose
# ("2 working days" written as "two working days", or a step numbered "1.")
# produces false positives that are cheaper to eyeball than to over-engineer
# away.

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Statutes spell numbers out ("forty-five days", "twenty-one days") while the
# model writes digits, so BOTH sides are normalised to digits before comparison.
# Compound tens must be handled before bare tens, or "forty-five" reads as "40".
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")[- ]+(" + "|".join(_UNITS) + r")\b",
    re.IGNORECASE,
)
_SIMPLE_RE = re.compile(
    r"\b(" + "|".join(list(_TENS) + list(_UNITS)) + r")\b", re.IGNORECASE
)

_UNIT_WORDS = r"(year|month|week|day|hour|rupee|per cent|percent|time)"

# Statutes interpose an adjective between the number and the unit ("twelve
# CALENDAR months", "two hundred and forty WORKING days"). Allowing one such
# word removes a whole class of false positives without loosening the match
# enough to pair unrelated numbers and units.
_FILLER = r"(?:\s+(?:calendar|working|clear|full|continuous|complete|completed))?"
_QUANTITY_RE = re.compile(
    r"\b(\d+)" + _FILLER + r"\s*" + _UNIT_WORDS + r"s?\b", re.IGNORECASE
)

# "two hundred and forty" -> 240. Written-out hundreds are common in Indian
# statutes ("one hundred and ninety days") and absent from the tens/units maps.
_HUNDREDS_RE = re.compile(
    r"\b(" + "|".join(_UNITS) + r")\s+hundred(?:\s+and)?"
    r"(?:\s+(" + "|".join(list(_TENS) + list(_UNITS)) + r"))?"
    r"(?:[- ]+(" + "|".join(_UNITS) + r"))?\b",
    re.IGNORECASE,
)


def _digits(text: str) -> str:
    """Rewrite spelled-out numbers as digits so both sides compare alike."""
    text = text.replace("per cent", "percent")

    def _hundreds(match: re.Match) -> str:
        total = _UNITS[match.group(1).lower()] * 100
        for group in (match.group(2), match.group(3)):
            if group:
                word = group.lower()
                total += _TENS.get(word) or _UNITS.get(word, 0)
        return str(total)

    def _compound(match: re.Match) -> str:
        return str(_TENS[match.group(1).lower()] + _UNITS[match.group(2).lower()])

    def _simple(match: re.Match) -> str:
        word = match.group(1).lower()
        return str(_TENS.get(word) or _UNITS[word])

    text = _HUNDREDS_RE.sub(_hundreds, text)
    return _SIMPLE_RE.sub(_simple, _COMPOUND_RE.sub(_compound, text))


def _quantities(text: str) -> set[tuple[str, str]]:
    """Normalised (amount, unit) pairs mentioned in `text`."""
    normalized = _digits(text)
    return {
        (match.group(1), match.group(2).lower().rstrip("s").replace(" ", ""))
        for match in _QUANTITY_RE.finditer(normalized)
    }


def check_numeric_claims(result: Result, state: Any, case: dict[str, Any]) -> list[str]:
    """Return advisory notes for quantities not traceable to a cited section."""
    verified = list(state.verified_section_ids or [])
    if not verified:
        return []

    # Strip the deterministic trailer/disclaimer — those are our own words.
    body = (state.final_answer or "").split("The law behind this answer:")[0]
    body = body.split("Verified citations:")[0]

    source_text = " ".join(
        CORPUS[s]["section_text"] + " " + CORPUS[s]["section_title"]
        for s in verified
        if s in CORPUS
    )
    source_quantities = _quantities(source_text) | _quantities(
        source_text.replace("-", " ")
    )
    query_quantities = _quantities(case["query"])

    notes: list[str] = []
    for amount, unit in sorted(_quantities(body)):
        if (amount, unit) in source_quantities or (amount, unit) in query_quantities:
            continue
        notes.append(f"{amount} {unit}(s)")
    return notes


# ---------------------------------------------------------------------------
# Per-query expectations
# ---------------------------------------------------------------------------

def check_expectations(result: Result, state: Any, case: dict[str, Any]) -> None:
    verified = list(state.verified_section_ids or [])
    retrieved = set(state.retrieval.section_ids) if state.retrieval else set()
    answer_lower = (state.final_answer or "").lower()
    grounded_lower = {c.lower() for c in (state.grounded_concepts or [])}

    expected_status = case.get("expect_status")
    if expected_status and state.status != expected_status:
        result.fail(
            f"status={state.status!r}, expected {expected_status!r}"
        )

    for concept in case.get("expect_concepts", []):
        if concept.lower() not in grounded_lower:
            result.fail(
                f"concept {concept!r} not grounded (got {sorted(grounded_lower)})"
            )

    for section_id in case.get("must_retrieve", []):
        if section_id not in retrieved:
            result.fail(f"{section_id!r} missing from the retrieval set")

    for section_id in case.get("must_cite", []):
        if section_id not in verified:
            result.fail(f"{section_id!r} not cited (verified: {verified})")

    # For questions where several provisions are independently defensible as
    # THE answer (e.g. a central Act and a state Act both governing the same
    # dismissal), assert that at least one was cited rather than pinning the
    # test to one particular correct answer.
    any_of = case.get("must_cite_any_of", [])
    if any_of and not any(section_id in verified for section_id in any_of):
        result.fail(
            f"none of {any_of} was cited (verified: {verified})"
        )

    for section_id in case.get("forbid_cite", []):
        if section_id in verified:
            result.fail(f"{section_id!r} was cited but is forbidden")

    for act_id in case.get("forbid_acts", []):
        offenders = [
            s for s in verified if act_registry.act_id_for_section(s) == act_id
        ]
        if offenders:
            result.fail(f"citations from forbidden act {act_id}: {offenders}")

    for needle in case.get("expect_text", []):
        if needle.lower() not in answer_lower:
            result.fail(f"answer missing expected text {needle!r}")

    for needle in case.get("forbid_text", []):
        if needle.lower() in answer_lower:
            result.fail(f"answer contains forbidden text {needle!r}")

    floor = case.get("min_confidence")
    if floor is not None and state.confidence < floor:
        result.fail(
            f"confidence {state.confidence:.2f} below floor {floor}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(case: dict[str, Any]) -> Result:
    result = Result(case_id=case["id"], query=case["query"])
    started = time.time()

    try:
        state = run(case["query"], case.get("persona", CITIZEN))
    except Exception as exc:  # noqa: BLE001
        result.elapsed = time.time() - started
        result.error = f"{type(exc).__name__}: {exc}"
        result.fail(f"PIPELINE RAISED — {result.error}")
        return result

    result.elapsed = time.time() - started
    result.status = state.status
    result.confidence = state.confidence
    result.grounded = list(state.grounded_concepts or [])
    result.verified = list(state.verified_section_ids or [])
    result.retrieved = sorted(state.retrieval.section_ids) if state.retrieval else []
    result.answer = state.final_answer or ""

    check_invariants(result, state, case)
    check_expectations(result, state, case)
    result.numeric_notes = check_numeric_claims(result, state, case)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="run only these ids")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with GOLDEN_PATH.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)["queries"]

    if args.only:
        wanted = {value.upper() for value in args.only}
        cases = [
            c for c in cases
            if c["id"].upper() in wanted
            or any(c["id"].upper().startswith(w) for w in wanted)
        ]

    if not cases:
        print("No matching cases.")
        raise SystemExit(1)

    print("=" * 78)
    print(f"Legal Graph RAG — verification harness  ({len(cases)} queries)")
    print("=" * 78)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, cases))
    total_elapsed = time.time() - started

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(
            f"[{mark}] {result.case_id:<26} {result.status:<22} "
            f"conf={result.confidence:.2f}  {result.elapsed:5.1f}s  "
            f"cites={len(result.verified)}"
        )
        for failure in result.failures:
            print(f"         !! {failure}")
        if args.verbose:
            print(f"         query:    {result.query}")
            print(f"         grounded: {result.grounded}")
            print(f"         verified: {result.verified}")
            for line in result.answer.splitlines():
                print(f"         | {line}")
            print()

    # ---- aggregate integrity metrics --------------------------------------
    all_citations = sum(len(r.verified) for r in results)
    hallucinated = sum(
        1 for r in results for s in r.verified if s not in CORPUS
    )
    repealed_cited = sum(
        1
        for r in results
        for s in r.verified
        if act_registry.act_id_for_section(s) in act_registry.REPEALED_ACT_IDS
    )

    flagged = [r for r in results if r.numeric_notes]

    print("=" * 78)
    print(f"PASSED {len(passed)}/{len(results)}   ({total_elapsed:.0f}s wall clock)")

    if flagged:
        print("-" * 78)
        print("ADVISORY — quantities in the prose not found in any cited section")
        print("(review by hand; numbers restated from the user's query are excluded)")
        for result in flagged:
            print(f"  {result.case_id}: {', '.join(result.numeric_notes)}")
            print(f"      cited: {result.verified}")

    print("-" * 78)
    print("Integrity metrics across the whole run:")
    print(f"  citations shown to users:        {all_citations}")
    print(f"  hallucinated (not in corpus):    {hallucinated}")
    print(f"  repealed law cited:              {repealed_cited}")
    print(f"  answers with unsourced numbers:  {len(flagged)}")
    print(
        f"  queries reaching a live answer:  "
        f"{sum(1 for r in results if r.status == 'ok')}/{len(results)}"
    )

    if failed:
        print("-" * 78)
        print("FAILING CASES:")
        for result in failed:
            print(f"  {result.case_id}: {result.query}")
            for failure in result.failures:
                print(f"      - {failure}")
        print("=" * 78)
        raise SystemExit(1)

    print("=" * 78)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
