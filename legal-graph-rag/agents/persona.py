"""
agents/persona.py

Persona handling for persona-aware answer synthesis.

The system serves two kinds of user, selected at login:
  - a Normal Citizen (no legal training) — wants reassurance and a plain,
    actionable explanation, and
  - a legal professional (Lawyer / Judge / Advocate) — wants a precise,
    technical, section-by-section treatment.

The persona changes ONLY how the answer is written (tone, technicality,
structure) and how the trailer is presented. It NEVER changes which sections
are retrieved, verified, or allowed to be cited — the integrity guarantees in
CLAUDE.md hold identically for both personas.

This module is the single source of truth for the persona vocabulary so the
synthesis node and the final-response node agree on the canonical values.
"""

from __future__ import annotations

# Canonical internal personas.
CITIZEN = "citizen"
LAWYER = "lawyer"

VALID_PERSONAS = (CITIZEN, LAWYER)

# Login-facing labels (and a couple of obvious synonyms) mapped onto the two
# canonical personas. Lawyer / Judge / Advocate all collapse to LAWYER.
_ALIASES = {
    "citizen": CITIZEN,
    "normal citizen": CITIZEN,
    "normal": CITIZEN,
    "user": CITIZEN,
    "public": CITIZEN,
    "lawyer": LAWYER,
    "judge": LAWYER,
    "advocate": LAWYER,
    "legal": LAWYER,
    "professional": LAWYER,
}


def match_persona(value: str | None) -> str | None:
    """
    Resolve a label to a canonical persona, or None if it is unknown. Use this
    when the caller needs to distinguish a recognised label from junk (e.g. a
    login prompt validating input).
    """
    if not value:
        return None
    return _ALIASES.get(value.strip().lower())


def normalize_persona(value: str | None) -> str:
    """
    Map an arbitrary persona label to a canonical persona. Anything unknown or
    missing falls back to CITIZEN — the safer default, since over-simplifying
    for a professional is less harmful than burying a layperson in legalese.
    """
    return match_persona(value) or CITIZEN
