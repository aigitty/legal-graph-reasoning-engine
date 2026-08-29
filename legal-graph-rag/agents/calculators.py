"""
agents/calculators.py

Deterministic entitlement calculator. NO LLM (CLAUDE.md rule 1 stays at
exactly three LLM calls — this module is pure Python arithmetic).

WHY THIS EXISTS
---------------
The pipeline could already tell a user "you are entitled to fifteen days' pay
for every year of service", but never "you are entitled to ₹57,692" — the
gap between describing a rule and answering the question someone actually
came with. Any generic chatbot can paraphrase a formula; doing the arithmetic
safely requires the formula to be verified against real statutory text, which
is exactly this project's core guarantee. This module is what turns that
guarantee into a number a user can act on.

WHY SOME OF THIS DELIBERATELY REFUSES TO COMPUTE A NUMBER
-----------------------------------------------------------
Gratuity (SSC_2020_S53) and notice pay (IRC_2020_S70/S79) are both
FULLY self-contained in the retrieved text — including, for gratuity, the
exact day-count divisor (Explanation 3: "dividing the monthly rate of wages
... by twenty-six and multiplying ... by fifteen"). Retrenchment
COMPENSATION ("fifteen days' average pay" per year, IRC_2020_S70(b)) is not:
"average pay" is defined in IRC_2020_S2(d) as an average of wages over the
preceding three months, with no stated rule for converting that monthly
average into a daily figure. Silently reusing the gratuity divisor would be
inventing a rule the retrieved text does not contain — the exact failure mode
CLAUDE.md rule 3 exists to prevent, just committed in Python instead of in an
LLM prompt. `retrenchment_compensation_gap()` says so honestly instead.

Each function computes from a fully-stated formula in one verified section,
never from unwritten convention. Where the corpus does not fully specify the
formula, the calculator says so rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Section ids this module knows how to compute from. Used by output_guardrail
# to verify each Calculation the same way an ordinary citation is verified
# (provenance + Neo4j existence) — a calculator result is still a claim about
# a section, and gets no exemption from CLAUDE.md rule 5 for being arithmetic
# rather than prose.
GRATUITY_SECTION = "SSC_2020_S53"
NOTICE_PAY_GENERAL_SECTION = "IRC_2020_S70"
NOTICE_PAY_CHAPTER_X_SECTION = "IRC_2020_S79"


def format_inr(amount: float) -> str:
    """
    Format a rupee amount with INDIAN digit grouping: 1234567 -> "₹12,34,567"
    (not the Western "₹1,234,567"). This is the grouping an Indian user
    actually expects; getting it wrong reads as obviously foreign-made.
    """
    n = round(amount)
    sign = "-" if n < 0 else ""
    digits = str(abs(n))

    if len(digits) <= 3:
        grouped = digits
    else:
        last3, rest = digits[-3:], digits[:-3]
        parts: list[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3

    return f"{sign}₹{grouped}"


@dataclass
class Calculation:
    """
    One entitlement, computed or honestly declined.

    `amount` is None when the retrieved text does not fully specify the
    formula (see module docstring) — `note` explains what is missing. A
    Calculation with `amount is None` is not a failure to be hidden; the gap
    itself is the honest answer and is shown to the user like any other
    "the law does not cover this" gap in the pipeline.
    """

    label: str
    section_id: str
    amount: float | None
    formula: str
    assumptions: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def is_computed(self) -> bool:
        return self.amount is not None


def _effective_years(years_of_service: float) -> int:
    """
    Completed years, plus one more if the remaining fraction exceeds six
    months — "for every completed year of continuous service or any part
    thereof in excess of six months", stated in near-identical wording in both
    SSC_2020_S53(2) (gratuity) and IRC_2020_S70(b) (retrenchment
    compensation). 4 years 8 months -> 8/12 = 0.667 > 0.5 -> 5 effective years.
    """
    whole = int(years_of_service)
    remainder = years_of_service - whole
    return whole + (1 if remainder > 0.5 else 0)


def compute_gratuity(monthly_salary: float, years_of_service: float) -> Calculation:
    """
    SSC_2020_S53(2) + Explanation 3.

    Explanation 3 gives the COMPLETE formula for a monthly-rated employee:
    daily wage = monthly salary / 26; fifteen days' wage = that × 15. This is
    the one entitlement in the corpus where every step of the arithmetic is
    stated in the retrieved text itself, which is why it is the flagship case.
    """
    eligible = years_of_service >= 5
    fifteen_day_wage = monthly_salary / 26 * 15

    if not eligible:
        return Calculation(
            label="Gratuity",
            section_id=GRATUITY_SECTION,
            amount=None,
            formula="(monthly salary ÷ 26 × 15) × completed years",
            note=(
                f"Gratuity under Section 53 requires at least 5 years of "
                f"continuous service; {years_of_service:g} year(s) does not "
                f"qualify on its face. Two exceptions in the retrieved text "
                f"could still apply: (1) the 5-year requirement is waived "
                f"entirely if employment ended due to death or disablement "
                f"(Section 53(1)); (2) if you worked at least 240 days in "
                f"what would be your final year, that year may count as a "
                f"full year of continuous service under the deeming rule in "
                f"Section 54, which could bring your total to 5 years even "
                f"though the calendar time is shorter. This calculator has no "
                f"way to check either from years of service alone, so it has "
                f"not assumed either applies — check with a labour authority "
                f"if you think you may qualify under (2)."
            ),
        )

    effective_years = _effective_years(years_of_service)
    amount = round(fifteen_day_wage * effective_years, 2)
    return Calculation(
        label="Gratuity",
        section_id=GRATUITY_SECTION,
        amount=amount,
        formula=(
            f"(₹{monthly_salary:,.0f} ÷ 26 × 15) × "
            f"{effective_years} year(s) = {format_inr(fifteen_day_wage)} "
            f"× {effective_years}"
        ),
        assumptions=[
            "Assumes a monthly-rated employee (Explanation 3 to Section 53) "
            "and that the given salary is the rate last drawn.",
            "Does not apply the statutory maximum cap on gratuity (Section "
            "53(3)): that cap amount is fixed by government notification, "
            "which is not present in the retrieved text.",
        ],
    )


def compute_notice_pay(
    monthly_salary: float, months: int, section_id: str
) -> Calculation:
    """
    IRC_2020_S70(a) (general, 1 month) / IRC_2020_S79(a) (Chapter X
    establishments, 3 months): "wages for the period of the notice" is a
    literal statement, not a formula requiring interpretation — monthly wage
    times the number of months of notice the section requires.
    """
    amount = round(monthly_salary * months, 2)
    period = f"{months} month" + ("s" if months != 1 else "")
    return Calculation(
        label=f"Notice pay ({period}, if not given working notice)",
        section_id=section_id,
        amount=amount,
        formula=f"₹{monthly_salary:,.0f} × {months} month(s)",
        assumptions=[
            f"Assumes your employer pays wages in lieu of notice rather than "
            f"giving you {months} month(s) of working notice — the Act "
            f"permits either."
        ],
    )


def retrenchment_compensation_gap(years_of_service: float) -> Calculation:
    """
    IRC_2020_S70(b): "fifteen days' average pay ... for every completed year".

    Deliberately returns NO amount. "Average pay" is defined in IRC_2020_S2(d)
    as an average of wages over the preceding three months (for a monthly-paid
    worker) — a MONTHLY figure, with no stated rule anywhere in the retrieved
    text for converting it into a daily rate for this "fifteen days" purpose.
    Gratuity has that conversion spelled out explicitly (Explanation 3);
    retrenchment compensation does not. Reusing the gratuity divisor here
    would be a plausible-looking guess with no textual basis — precisely what
    this calculator exists to avoid doing.
    """
    effective_years = _effective_years(years_of_service)
    return Calculation(
        label="Retrenchment compensation",
        section_id=NOTICE_PAY_GENERAL_SECTION,
        amount=None,
        formula="fifteen days' average pay × completed years of service",
        note=(
            f"Section 70(b) entitles you to fifteen days' average pay for "
            f"each of your {effective_years} effective year(s) of service, "
            f"but the retrieved text does not state how to convert your "
            f"average pay into a daily figure for this calculation (unlike "
            f"gratuity, where the divide-by-26 rule is given explicitly). We "
            f"have not guessed at that conversion, so no rupee figure is "
            f"given here — ask a labour authority for the daily-rate "
            f"convention actually used."
        ),
    )


def compute_available(
    grounded_concepts: list[str],
    monthly_salary: float | None,
    years_of_service: float | None,
    retrieved_section_ids: set[str],
) -> list[Calculation]:
    """
    Every Calculation this query's facts and grounded concepts support.

    Gated on `retrieved_section_ids` (provenance): a Calculation is only
    offered when its section was actually retrieved for this query, mirroring
    the citation rule that the model may only cite what it was shown — the
    calculator gets no exemption from that rule for being code instead of a
    model. Gated on the numeric inputs being PRESENT: this never blocks an
    answer for a missing fact (CLAUDE.md section 7); it simply omits the
    calculation when the salary or tenure was never given, exactly as
    years_of_service already works elsewhere in the pipeline.
    """
    if not monthly_salary or monthly_salary <= 0:
        return []

    concepts = set(grounded_concepts)
    calculations: list[Calculation] = []

    if (
        "gratuity" in concepts
        and years_of_service is not None
        and GRATUITY_SECTION in retrieved_section_ids
    ):
        calculations.append(compute_gratuity(monthly_salary, years_of_service))

    termination_concepts = {
        "retrenchment",
        "wrongful termination",
        "retrenchment compensation",
        "notice period",
    }
    if termination_concepts & concepts:
        if NOTICE_PAY_GENERAL_SECTION in retrieved_section_ids:
            calculations.append(
                compute_notice_pay(monthly_salary, 1, NOTICE_PAY_GENERAL_SECTION)
            )
        if NOTICE_PAY_CHAPTER_X_SECTION in retrieved_section_ids:
            calculations.append(
                compute_notice_pay(
                    monthly_salary, 3, NOTICE_PAY_CHAPTER_X_SECTION
                )
            )
        if (
            years_of_service is not None
            and NOTICE_PAY_GENERAL_SECTION in retrieved_section_ids
        ):
            calculations.append(retrenchment_compensation_gap(years_of_service))

    return calculations
