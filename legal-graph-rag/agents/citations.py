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

# THE SECOND BUG THIS FIXES
# -------------------------
# The replacement pattern matched exactly ONE id per bracket, so a GROUPED
# marker — which the lawyer persona produces constantly when several provisions
# bear on one proposition —
#
#     [IRC_2020_S43, IRC_2020_S44, IRC_2020_S49, IRC_2020_S53]
#     [IRC_2020_S99(1), IRC_2020_S99(2)(zp)]
#
# matched nothing at all: after "IRC_2020_S43" the next character is a comma
# rather than the "]" the pattern required. That reopened the exact hole this
# module was written to close — those ids were never parsed, so never verified,
# so never stripped. Observed live: IRC_2020_S99 and IRC_2020_S49 reached the
# user inside a citation-looking marker while `verified_section_ids` listed
# neither. A grouped marker is now parsed id-by-id and re-emitted containing
# only the ids that actually survived verification.
_SECTION_ID = r"[A-Z0-9][A-Z0-9_/.\-]{2,}"
_SUBSECTION = r"(?:\s*\([0-9A-Za-z]{1,8}\))*"
_ONE_CITATION = rf"{_SECTION_ID}{_SUBSECTION}"

# Separators tolerated INSIDE one bracket: comma, semicolon, "and", or plain
# whitespace. Deliberately generous — the format is the model's choice, and this
# module's whole purpose is that an unrecognised marker is an unverified one.
# Being strict here does not stop the model writing "[A and B]"; it only stops
# us from checking it. Each part is still required to look like a section id
# (see _base_ids), so a bracketed label such as [PRIMARY] is untouched.
_SEPARATOR = r"(?:\s*[,;]\s*|\s+(?:and\s+)?)"

# A citation marker: one or more section ids in a single bracket, each
# optionally followed by parenthesised sub-section parts — (1), (3A), (k), (1)(a).
#   group 1 = the whole inner list, e.g. "IDA_1947_S7(1), IDA_1947_S9"
CITATION_RE = re.compile(
    rf"\[({_ONE_CITATION}(?:{_SEPARATOR}{_ONE_CITATION})*)\]"
)

# Splits one bracket's contents into individual citations.
#   group 1 = base section id      e.g. "IDA_1947_S7"
#   group 2 = sub-section suffix   e.g. "(1)(a)"  (may be empty)
CITATION_PART_RE = re.compile(rf"({_SECTION_ID})({_SUBSECTION})")


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
        for token in _base_ids(match.group(1)):
            if token not in seen:
                seen.add(token)
                ordered.append(token)
    return ordered


def _base_ids(inner: str) -> list[str]:
    """
    Base section ids inside one bracket's contents, in order.

    Returns [] unless EVERY part looks like a section id, so a bracket that is
    really a label — [PRIMARY], [SUPPORTING] — is never treated as a citation
    group and is left untouched by the renderer.
    """
    parts = [m.group(1) for m in CITATION_PART_RE.finditer(inner)]
    if not parts or not all(looks_like_citation(part) for part in parts):
        return []
    return parts


def _clean_spacing(text: str) -> str:
    """
    Tidy the gaps a removed marker leaves behind.

    Removing a marker does not just leave a double space — it can strand the
    punctuation that separated it from its neighbours. A sentence written as

        "...sickness, accident [X], [Y], or strikes [Z]."

    collapses to "...sickness, accident,, or strikes." and then ",." at the end.
    The citizen persona strips EVERY marker, so it hits this on almost any
    answer that cites two provisions in one clause; both artifacts were visible
    in live output ("lock-outs,." and "in most cases,.").

    So: drop spaces before punctuation, collapse runs of punctuation left
    adjacent by a removal (keeping the strongest terminator), remove a dangling
    conjunction before a full stop, and squeeze whitespace.
    """
    text = re.sub(r"\(\s*\)", "", text)

    # Applied repeatedly to a fixed point: one rule's output is another's input.
    # "see [X], and [Y]." strips to "see, and." — the conjunction rule then
    # leaves "see,." which only the comma rule can finish. A single pass in any
    # order fixes one of the two and not the other.
    rules = (
        (r"[ \t]+([,.;:!?)\]])", r"\1"),          # space before punctuation
        (r"\s*\b(?:and|or)\b\s*([.;:!?])", r"\1"),  # dangling conjunction
        (r",\s*([.;:!?])", r"\1"),                # ",." -> "."
        (r",(\s*,)+", ","),                       # ",,"  -> ","
        (r"([.;:!?])\s*\1+", r"\1"),              # ".."  -> "."
    )
    for _ in range(4):
        before = text
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if text == before:
            break

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

    A GROUPED marker is filtered part by part rather than kept or dropped whole:
    [S43, S44, S49] where S49 failed verification is re-emitted as [S43, S44].
    Dropping the group entirely would discard two good citations for one bad
    one; keeping it whole would leak the bad one, which is the failure this
    guards against.
    """

    def _replace(match: re.Match) -> str:
        inner = match.group(1)
        parts = [
            (m.group(1), m.group(0)) for m in CITATION_PART_RE.finditer(inner)
        ]
        if not parts or not all(looks_like_citation(base) for base, _ in parts):
            return match.group(0)  # not a citation marker — leave it alone

        kept = [whole.strip() for base, whole in parts if base in verified]
        if not kept or drop_verified:
            return ""
        return "[" + ", ".join(kept) + "]"

    return _clean_spacing(CITATION_RE.sub(_replace, text))
