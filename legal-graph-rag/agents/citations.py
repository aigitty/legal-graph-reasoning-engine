"""
agents/citations.py

Single source of truth for the inline citation-marker format.

WHY THIS MODULE EXISTS
----------------------
The marker regex used to be duplicated in two files — synthesis_node.py (which
PARSES markers so they can be verified) and final_response_node.py (which STRIPS
the ones that failed verification). Two copies of the rule that decides what
counts as a citation is one copy too many: anything the parser fails to
recognise is never verified, and if the stripper fails to recognise it too, it
sails through to the user unchecked. That is precisely the hole this module was
created to close.

THE BUG THIS FIXES
------------------
The old pattern allowed only [A-Z0-9_/.-] inside the brackets. Lawyer-persona
answers legitimately cite sub-sections:

    [IDA_1947_S7(1)]      [IDA_1947_S7A(3)]      [IDA_1947_S2(k)]

The parentheses fall outside that character class, so the marker matched
NEITHER regex. The result: those citations were never parsed, so never verified,
so never stripped — an entire answer's worth of unverified section references
reaching the user, while `verified_section_ids` reported a single citation. The
integrity guarantee silently did not apply to the exact citation style a
practitioner is most likely to use.

Sub-section references are legitimate and worth keeping. The rule is that the
BASE section id is what gets verified: [IDA_1947_S7(1)] is admissible if and
only if IDA_1947_S7 was in the evidence pack and exists in Neo4j. The
sub-section suffix is presentation detail on top of a verified section.
"""

from __future__ import annotations

import re

# A citation marker: a section id, optionally followed by one or more
# parenthesised sub-section parts — (1), (3A), (k), (1)(a) — inside brackets.
#   group 1 = base section id      e.g. "IDA_1947_S7"
#   group 2 = sub-section suffix   e.g. "(1)(a)"  (may be empty)
CITATION_RE = re.compile(
    r"\[([A-Z0-9][A-Z0-9_/.\-]{2,})((?:\s*\([0-9A-Za-z]{1,8}\))*)\]"
)


def looks_like_citation(token: str) -> bool:
    """
    True if `token` looks like a section id rather than a label.

    Real section ids carry a digit, an underscore, or a slash; this excludes
    bare bracketed words such as [PRIMARY] or [SUPPORTING], which the evidence
    pack format uses and which must be left alone.
    """
    return any(ch.isdigit() for ch in token) or "_" in token or "/" in token


def parse_citations(text: str) -> list[str]:
    """
    Extract BASE section ids from the markers in `text`, in first-cited order,
    deduplicated.

    Sub-section suffixes are deliberately discarded here: verification is a
    question about the section, and the graph stores sections, not sub-sections.
    No filtering or existence checking happens in this function — that belongs
    to output_guardrail_node.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for match in CITATION_RE.finditer(text):
        token = match.group(1)
        if looks_like_citation(token) and token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _clean_spacing(text: str) -> str:
    """Tidy the gaps a removed marker leaves behind (" ," -> ",", double spaces)."""
    text = re.sub(r"[ \t]+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def render_markers(text: str, verified: set[str], drop_verified: bool) -> str:
    """
    Rewrite inline citation markers according to what survived verification.

    Unverified markers are ALWAYS removed — that is the rule-5 guarantee and it
    does not depend on persona. `drop_verified` additionally removes the ones
    that passed, which is what the citizen persona wants (they have already been
    checked upstream; a machine id is just noise to a layperson).

    A marker is judged by its BASE section id, so [IDA_1947_S7(1)] survives if
    and only if IDA_1947_S7 was verified. Non-citation brackets such as
    [PRIMARY] are left untouched.
    """

    def _replace(match: re.Match) -> str:
        token = match.group(1)
        if not looks_like_citation(token):
            return match.group(0)  # not a citation marker — leave it alone
        if token not in verified:
            return ""
        return "" if drop_verified else match.group(0)

    return _clean_spacing(CITATION_RE.sub(_replace, text))
