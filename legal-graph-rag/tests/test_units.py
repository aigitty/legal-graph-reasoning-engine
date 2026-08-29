"""
tests/test_units.py

FAST OFFLINE unit layer. No Neo4j, no Vertex AI, no network.

    python -m tests.test_units

tests/verify.py is the end-to-end harness: it runs all 55 golden queries
against real Neo4j and real Gemini and takes ~11 minutes. That is the right
tool for "does the pipeline still answer correctly", and the wrong tool for
"did I break the citation regex" — a question that can be settled in
milliseconds and that no one wants to wait 11 minutes to ask.

Everything covered here is deterministic logic on the integrity path:

  agents/citations.py  — the rule-5 gate. Both live bugs this module has had
                         were regex bugs that let UNVERIFIED section ids reach
                         the user, and both were invisible until someone read
                         an answer closely. They are exactly the kind of thing
                         a unit test catches for free.
  agents/ontology.py   — grounding and the companion (remedy) layer.
  graph/act_registry.py — the temporal helpers behind the commencement line.

Exit code is non-zero if anything fails, so it can gate a commit.
"""

from __future__ import annotations

import sys
from datetime import date

from agents.calculators import (
    GRATUITY_SECTION,
    NOTICE_PAY_CHAPTER_X_SECTION,
    NOTICE_PAY_GENERAL_SECTION,
    compute_available,
    compute_gratuity,
    compute_notice_pay,
    format_inr,
    retrenchment_compensation_gap,
)
from agents.citations import parse_citations, render_markers
from agents.ontology import companion_concepts, ground_query
from graph import act_registry

_FAILURES: list[str] = []


def check(name: str, got: object, want: object) -> None:
    if got == want:
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(name)
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def check_true(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        _FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# agents/citations.py
# ---------------------------------------------------------------------------

def test_citations() -> None:
    print("\nagents/citations.py")

    check("parse plain", parse_citations("see [IRC_2020_S43]"), ["IRC_2020_S43"])
    check("parse sub-section", parse_citations("[IDA_1947_S7(1)]"), ["IDA_1947_S7"])
    check("parse nested sub", parse_citations("[IDA_1947_S7(1)(a)]"), ["IDA_1947_S7"])
    check("parse dedups", parse_citations("[A_1_S1] [A_1_S1(2)]"), ["A_1_S1"])
    check("parse ignores label", parse_citations("#1 [PRIMARY] x"), [])

    # Grouped markers. These matched NOTHING before — the pattern required "]"
    # immediately after the first id — so every id in a group went unparsed,
    # unverified and unstripped straight to the user.
    check(
        "parse grouped",
        parse_citations("[IRC_2020_S43, IRC_2020_S44]"),
        ["IRC_2020_S43", "IRC_2020_S44"],
    )
    check(
        "parse grouped with sub-sections",
        parse_citations("[IRC_2020_S99(1), IRC_2020_S99(2)(zp)]"),
        ["IRC_2020_S99"],
    )
    check(
        "parse grouped with 'and'",
        parse_citations("[IRC_2020_S43 and IRC_2020_S44]"),
        ["IRC_2020_S43", "IRC_2020_S44"],
    )
    check(
        "parse grouped with semicolon",
        parse_citations("[IRC_2020_S43; IRC_2020_S44]"),
        ["IRC_2020_S43", "IRC_2020_S44"],
    )
    check(
        "parse grouped bare whitespace",
        parse_citations("[IRC_2020_S43 IRC_2020_S44]"),
        ["IRC_2020_S43", "IRC_2020_S44"],
    )

    verified = {"IRC_2020_S43", "IRC_2020_S53", "IDA_1947_S7"}

    check(
        "render strips unverified",
        render_markers("a [IRC_2020_S99] b", verified, drop_verified=False),
        "a b",
    )
    check(
        "render keeps verified (lawyer)",
        render_markers("a [IRC_2020_S43] b", verified, drop_verified=False),
        "a [IRC_2020_S43] b",
    )
    check(
        "render drops verified (citizen)",
        render_markers("a [IRC_2020_S43] b", verified, drop_verified=True),
        "a b",
    )
    check(
        "render keeps verified sub-section",
        render_markers("a [IDA_1947_S7(1)] b", verified, drop_verified=False),
        "a [IDA_1947_S7(1)] b",
    )
    check(
        "render leaves non-citation label alone",
        render_markers("a [PRIMARY] b", verified, drop_verified=False),
        "a [PRIMARY] b",
    )

    # A group is filtered part by part: dropping it whole would lose two good
    # citations for one bad one, keeping it whole would leak the bad one.
    check(
        "render filters group",
        render_markers(
            "a [IRC_2020_S43, IRC_2020_S49, IRC_2020_S53] b",
            verified,
            drop_verified=False,
        ),
        "a [IRC_2020_S43, IRC_2020_S53] b",
    )
    check(
        "render drops fully-unverified group",
        render_markers("a [IRC_2020_S98, IRC_2020_S99] b", verified, drop_verified=False),
        "a b",
    )

    # Punctuation stranded by a removal. The citizen persona strips EVERY
    # marker, so this hits almost any answer citing two provisions in one
    # clause. Live output showed "lock-outs,." and "in most cases,.".
    check(
        "strips punctuation stranded by removal",
        render_markers(
            "sickness, accident [IRC_2020_S43], [IRC_2020_S99], or strikes "
            "[IRC_2020_S98].",
            verified,
            drop_verified=True,
        ),
        "sickness, accident, or strikes.",
    )
    check(
        "strips dangling conjunction",
        render_markers("see [IRC_2020_S99], and [IRC_2020_S98].", verified,
                       drop_verified=True),
        "see.",
    )
    check(
        "leaves ordinary prose punctuation alone",
        render_markers("you may appeal, and the tribunal decides.", verified,
                       drop_verified=True),
        "you may appeal, and the tribunal decides.",
    )

    # THE INVARIANT: after rendering, no unverified id survives anywhere in the
    # visible text, in any marker shape.
    draft = (
        "x [IRC_2020_S43] y [IRC_2020_S99] z [IRC_2020_S43, IRC_2020_S99] "
        "w [IRC_2020_S99(2)(zp)] v [IDA_1947_S7(1)]"
    )
    rendered = render_markers(draft, verified, drop_verified=False)
    check_true(
        "no unverified id survives rendering",
        "IRC_2020_S99" not in rendered,
        f"rendered={rendered!r}",
    )


# ---------------------------------------------------------------------------
# agents/ontology.py
# ---------------------------------------------------------------------------

def test_grounding() -> None:
    print("\nagents/ontology.py — grounding")

    # The verb/noun split. "retrench" is not a substring of "retrenchment" and
    # shares no raw token with it, so this query used to ground ONLY to
    # `compensation for injury at work` (matched on the stray word "workmen").
    check_true(
        "verb form grounds the noun concept",
        "retrenchment" in ground_query("in what order must an employer retrench workmen?"),
        f"got {ground_query('in what order must an employer retrench workmen?')}",
    )
    check_true(
        "past tense grounds too",
        "retrenchment" in ground_query("I was retrenched last month"),
    )
    check_true(
        "gratuity still grounds",
        "gratuity" in ground_query("am I owed gratuity after 4 years?"),
    )
    # A generic English word must not carry a concept on its own. The alias
    # "what happens to the employer" reduces to the single token "happen",
    # which is rare across the vocabulary and so passed the distinctiveness
    # guard with a perfect score — every "what happens to..." question grounded
    # to employer penalties.
    takeover = ground_query(
        "my company was taken over by another firm, what happens to my service?"
    )
    check_true(
        "'what happens' does not ground penalties",
        "penalties for employer offences" not in takeover,
        f"got {takeover}",
    )
    check_true(
        "takeover grounds transfer of undertaking",
        "transfer of undertaking" in takeover,
        f"got {takeover}",
    )
    # Stage 1 is not stopworded, so the verbatim alias still grounds.
    check_true(
        "verbatim alias still grounds penalties",
        "penalties for employer offences"
        in ground_query("what happens to the employer if they break the law?"),
    )

    check("out-of-domain grounds nothing", ground_query("how do I file for divorce?"), [])
    check("nonsense grounds nothing", ground_query("asdkjasjdas qqq"), [])


def test_companion_concepts() -> None:
    print("\nagents/ontology.py — companion (remedy) layer")

    termination = companion_concepts(["retrenchment"])
    check_true(
        "termination query reaches a forum",
        "labour court and tribunal" in termination,
        f"got {termination}",
    )
    wages = companion_concepts(["salary delay"])
    check_true(
        "wage query reaches wage recovery",
        "recovering unpaid wages" in wages,
        f"got {wages}",
    )
    check(
        "already-grounded companions are not duplicated",
        companion_concepts(["salary delay", "recovering unpaid wages"]),
        ["appeal against an order"],
    )
    check("nothing grounded means no companions", companion_concepts([]), [])


# ---------------------------------------------------------------------------
# graph/act_registry.py — temporal helpers
# ---------------------------------------------------------------------------

def test_act_registry() -> None:
    print("\ngraph/act_registry.py — temporal helpers")

    check("format_date", act_registry.format_date("2025-11-21"), "21 November 2025")
    check("format_date empty", act_registry.format_date(""), "")
    check("format_date malformed passes through", act_registry.format_date("xx"), "xx")
    check("commencement_date IRC", act_registry.commencement_date("IRC_2020"), "21 November 2025")

    notes = act_registry.commencement_notes(["IRC_2020", "IDA_1947", "NOT_AN_ACT"])
    check("commencement_notes excludes repealed and unknown", notes,
          [("Industrial Relations Code, 2020", "21 November 2025")])

    check_true(
        "replacement_note names successor and date",
        act_registry.replacement_note("IDA_1947").startswith(
            "The Industrial Disputes Act, 1947 was replaced by the "
            "Industrial Relations Code, 2020 on 21 November 2025"
        ),
        act_registry.replacement_note("IDA_1947"),
    )
    check("replacement_note empty for in-force act",
          act_registry.replacement_note("IRC_2020"), "")

    # The extraction prompt's corpus list is built from this; if it stops
    # naming the operative Codes the first LLM call reasons about the wrong
    # corpus, which is how it went stale the first time.
    block = act_registry.corpus_block()
    for act_name in ("Industrial Relations Code, 2020", "Code on Social Security, 2020",
                     "Code on Wages, 2019"):
        check_true(f"corpus_block names {act_name}", act_name in block)
    check_true("corpus_block marks repealed acts",
               "REPEALED" in block and "Industrial Disputes Act, 1947" in block)


def test_temporal_mismatch() -> None:
    print("\ngraph/act_registry.py — pre-commencement (temporal mismatch) check")

    check("parse full date", act_registry.parse_flexible_date("2025-08-03"),
          date(2025, 8, 3))
    check("parse month precision", act_registry.parse_flexible_date("2025-08"),
          date(2025, 8, 1))
    check("parse year precision", act_registry.parse_flexible_date("2025"),
          date(2025, 1, 1))
    check("parse None", act_registry.parse_flexible_date(None), None)
    check("parse garbage", act_registry.parse_flexible_date("not a date"), None)

    check_true(
        "event before commencement is flagged",
        act_registry.event_predates_act(date(2025, 8, 15), "IRC_2020"),
    )
    check_true(
        "event after commencement is not flagged",
        not act_registry.event_predates_act(date(2025, 12, 1), "IRC_2020"),
    )
    check_true(
        "event ON commencement day is not flagged",
        not act_registry.event_predates_act(date(2025, 11, 21), "IRC_2020"),
    )

    # The end-to-end case this whole mechanism exists for: "I was retrenched in
    # August 2025" grounds to the IRC (in force from 21 Nov 2025) and must be
    # flagged, naming the Act it replaced.
    conflicts = act_registry.commencement_conflicts("2025-08", {"IRC_2020"})
    check_true(
        "August 2025 vs IRC_2020 is a conflict naming the predecessor",
        conflicts == [("Industrial Relations Code, 2020", "21 November 2025",
                       "Industrial Disputes Act, 1947")],
        f"got {conflicts}",
    )
    check("no event date means no conflicts",
          act_registry.commencement_conflicts(None, {"IRC_2020"}), [])
    check("no conflict when event postdates commencement",
          act_registry.commencement_conflicts("2026-01", {"IRC_2020"}), [])
    check("act with no commencement conflict is skipped",
          act_registry.commencement_conflicts("2020-01", {"KSEA_1961"}), [])


def test_calculators() -> None:
    print("\nagents/calculators.py — entitlement calculator")

    check("INR grouping under 1000", format_inr(692), "₹692")
    check("INR grouping thousands", format_inr(57692), "₹57,692")
    check("INR grouping lakhs", format_inr(1234567), "₹12,34,567")
    check("INR grouping rounds", format_inr(999.6), "₹1,000")
    check("INR grouping negative", format_inr(-500), "-₹500")

    # 5 years 8 months: ALREADY past the 5-year eligibility threshold, so the
    # "or part thereof in excess of six months" rule (SSC_2020_S53(2)) rounds
    # the amount up to 6 effective years. 8/12 = 0.667 > 0.5 -> rounds up.
    g = compute_gratuity(25000.0, 5 + 8 / 12)
    check_true("gratuity is computed once past the 5-year threshold", g.is_computed)
    check("gratuity effective-year rounding gives 6 years worth",
          round(g.amount, 2), round(25000 / 26 * 15 * 6, 2))
    check("gratuity cites the real section", g.section_id, GRATUITY_SECTION)

    # Exactly 6 months does NOT exceed six months — "in excess of" is strict —
    # so 5 years 6 months stays at 5 effective years, not 6.
    g_exact_half = compute_gratuity(25000.0, 5.5)
    check("exactly half a year does not round up",
          round(g_exact_half.amount, 2), round(25000 / 26 * 15 * 5, 2))

    # 4 years 8 months is BELOW the 5-year threshold on its face. This is
    # deliberately NOT computed, even though it looks close: SSC_2020_S54 has
    # a 240-day continuous-service deeming rule that could make this person
    # eligible after all, and this calculator has no way to check that from
    # years_of_service alone. Refusing to guess here — in either direction —
    # is the correct behaviour, not a bug: a false "yes you qualify" is exactly
    # as harmful as a false "no you don't".
    g_near_miss = compute_gratuity(25000.0, 4 + 8 / 12)
    check_true("under-5-years near-miss is not silently assumed eligible",
               not g_near_miss.is_computed)
    check_true("near-miss note names the 240-day exception",
               "240 days" in g_near_miss.note, g_near_miss.note)

    # Comfortably under 5 years: no realistic ambiguity, still refuses a number.
    g_ineligible = compute_gratuity(25000.0, 3.0)
    check_true("well under 5 years has no computed amount", not g_ineligible.is_computed)
    check_true("ineligibility reason mentions 5 years",
               "5 years" in g_ineligible.note, g_ineligible.note)

    # Notice pay is a flat multiply — no rounding logic to get wrong.
    n1 = compute_notice_pay(30000.0, 1, NOTICE_PAY_GENERAL_SECTION)
    check("one month notice pay", n1.amount, 30000.0)
    n3 = compute_notice_pay(30000.0, 3, NOTICE_PAY_CHAPTER_X_SECTION)
    check("three months notice pay", n3.amount, 90000.0)

    # THE INTEGRITY CASE: retrenchment "compensation" must NEVER return a
    # number, because the corpus defines "average pay" (IRC_2020_S2(d)) as a
    # monthly average with no stated day-count conversion — unlike gratuity's
    # explicit divide-by-26 (Explanation 3 to SSC_2020_S53). Silently reusing
    # the gratuity divisor here would be exactly the fabrication this whole
    # project exists to prevent, just committed in Python instead of a prompt.
    gap = retrenchment_compensation_gap(4 + 8 / 12)
    check_true("retrenchment compensation refuses to guess a number",
               not gap.is_computed)
    check_true("gap explains what is missing",
               "average pay" in gap.note.lower(), gap.note)

    # compute_available: provenance gating. A calculation must never appear
    # for a section that was not actually retrieved for this query — the
    # calculator gets no exemption from the citation provenance rule for being
    # arithmetic instead of prose.
    all_sections = {GRATUITY_SECTION, NOTICE_PAY_GENERAL_SECTION,
                     NOTICE_PAY_CHAPTER_X_SECTION}
    calcs = compute_available(["gratuity"], 25000.0, 4 + 8 / 12, all_sections)
    check("gratuity concept yields exactly one calculation", len(calcs), 1)
    check("compute_available omits it when section wasn't retrieved",
          compute_available(["gratuity"], 25000.0, 5.0, set()), [])
    check("compute_available needs a salary to do anything",
          compute_available(["gratuity"], None, 5.0, all_sections), [])
    check_true(
        "retrenchment concept yields notice pay + the honest gap, not a guess",
        {c.label for c in compute_available(
            ["retrenchment"], 30000.0, 4 + 8 / 12, all_sections
        )} == {
            "Notice pay (1 month, if not given working notice)",
            "Notice pay (3 months, if not given working notice)",
            "Retrenchment compensation",
        },
    )
    check(
        "unrelated concept (no salary-relevant grounding) computes nothing",
        compute_available(["annual leave"], 30000.0, 5.0, all_sections), [],
    )


def main() -> int:
    test_citations()
    test_grounding()
    test_companion_concepts()
    test_act_registry()
    test_temporal_mismatch()
    test_calculators()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S): " + ", ".join(_FAILURES))
        return 1
    print("ALL UNIT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
