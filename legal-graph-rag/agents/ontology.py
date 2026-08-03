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
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")

CONCEPT_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ontology" / "concept_map.json"
)

FUZZY_CUTOFF = 0.8

# Longest word n-gram compared against the concept vocabulary in the fuzzy
# fallback. 4 comfortably spans the longest multi-word concept names
# ("re-employment of retrenched workmen" tokenises shorter than this once the
# query is normalised) without quadratic blow-up on a long query.
MAX_FUZZY_NGRAM = 4

# Ignore very short n-grams: at 3 characters difflib starts matching noise
# ("pay" against "day"), which grounds concepts the user never asked about.
MIN_FUZZY_CHARS = 5

# How many close matches to accept from a single n-gram. More than one is
# allowed because a phrase can legitimately sit between two concepts, but the
# loop stops at the first n-gram that matches at all.
MAX_FUZZY_MATCHES = 2

# Token-overlap threshold for the paraphrase stage. 0.5 means "half the phrase
# is accounted for by this concept's vocabulary, or half of one of its terms is
# accounted for by the phrase".
OVERLAP_CUTOFF = 0.5

# Cap on concepts grounded by the paraphrase stage alone. A vague phrase can
# overlap many concepts; taking the best few keeps retrieval focused while still
# letting a genuinely two-sided phrase ground both sides.
MAX_OVERLAP_MATCHES = 3

# Words carrying no discriminating power when grounding a short legal phrase.
# Deliberately includes generic legal filler ("law", "act", "rule") — a phrase
# grounding on the word "act" alone would match nothing useful.
_GROUNDING_STOPWORDS = frozenset(
    """
    the and or of in on at to for from by with without under over into about
    is are was were be been being do does did have has had can could shall
    should will would may might must not no nor if then than that this these
    those what which who whom when where why how any all some such same other
    my me you your our their his her its it they them we he she
    act acts law laws legal rule rules section sections code codes provision
    provisions employer employee employees employer's company companies work
    working job get got give given take taken make made need want know tell
    please help question situation case matter thing status procedure process
    file filing apply applying seek seeking go going put putting use using
    next done doing entitled entitlement right rights valid validity
    """.split()
)

# A single shared token only grounds a concept if the token is DISTINCTIVE —
# present in at most this many concepts. "dispute" appears across industrial
# dispute, conciliation, tribunals and arbitration, so "property dispute"
# must not ground on it alone; "appeal" or "gratuity" appear in exactly one
# concept each and are safe.
MAX_DF_FOR_SINGLE_TOKEN = 2


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


def _norm_token(token: str) -> str:
    """Crude singularisation so 'wages'/'wage' and 'deductions'/'deduction' meet."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    """Meaningful tokens of a phrase: no stopwords, no 1-2 char noise."""
    return {
        _norm_token(token)
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _GROUNDING_STOPWORDS
    }


# Precomputed token sets for every concept name and alias, built once at import
# so grounding stays a dictionary lookup rather than a per-query rescan.
_TERM_TOKENS: list[tuple[frozenset[str], str]] = [
    (frozenset(tokens), concept.name)
    for concept in CONCEPTS
    for term in concept.match_terms
    if (tokens := _content_tokens(term))
]

# Document frequency of each token = how many distinct CONCEPTS use it anywhere
# in their name or aliases. Drives the distinctiveness guard in _overlap_matches.
_TOKEN_DF: dict[str, int] = {}
for _concept in CONCEPTS:
    _seen_tokens: set[str] = set()
    for _term in _concept.match_terms:
        _seen_tokens |= _content_tokens(_term)
    for _token in _seen_tokens:
        _TOKEN_DF[_token] = _TOKEN_DF.get(_token, 0) + 1


def _overlap_matches(phrase: str) -> list[str]:
    """
    Concepts whose vocabulary overlaps `phrase` strongly enough to ground it.

    WHY THIS STAGE EXISTS
    ---------------------
    Substring matching only fires when an alias appears VERBATIM inside the
    phrase. The extraction LLM does not write aliases — it writes its own
    paraphrase. Real observed misses, every one of which is an obviously
    correct grounding a human would make instantly:

        "appeal"                -> concept "appeal against an order"
        "workman definition"    -> concept "who counts as an employee or workman"
        "wage claim"            -> concept "recovering unpaid wages"
        "validity of agreement" -> concept "agreements that waive legal rights"

    Each failed because the alias is LONGER than the phrase, so it cannot be a
    substring of it. Scoring shared tokens in both directions fixes that class
    outright instead of chasing it with ever more aliases.

    The score is symmetric — max(share of the phrase covered, share of the term
    covered) — because either direction is evidence: a phrase that is entirely
    contained in a concept term ("appeal") is as strong a signal as a term
    entirely contained in a long phrase ("punishment for non-payment of wages"
    against alias "punishment for employer").

    Being generous here is the right trade. Over-matching costs a few extra
    sections that graph/ranking.py then sorts below the ones that matter and the
    cap discards; under-matching costs the user an answer entirely.
    """
    phrase_tokens = _content_tokens(phrase)
    if not phrase_tokens:
        return []

    best: dict[str, float] = {}
    for term_tokens, concept_name in _TERM_TOKENS:
        shared = phrase_tokens & term_tokens
        if not shared:
            continue
        # Require at least one substantial shared token, so concepts are never
        # grounded on an incidental short word alone.
        if not any(len(token) >= 4 for token in shared):
            continue
        # One shared token is only enough when that token is distinctive.
        # Without this, "property dispute" grounds to `industrial dispute` on
        # the word "dispute" and the engine answers a property question with
        # labour law.
        if len(shared) == 1:
            token = next(iter(shared))
            if _TOKEN_DF.get(token, 0) > MAX_DF_FOR_SINGLE_TOKEN:
                continue
        score = max(len(shared) / len(phrase_tokens), len(shared) / len(term_tokens))
        if score >= OVERLAP_CUTOFF and score > best.get(concept_name, 0.0):
            best[concept_name] = score

    return [name for name, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def _ngrams(query: str, max_n: int = MAX_FUZZY_NGRAM) -> list[str]:
    """
    Word n-grams of `query`, longest first.

    Longest-first matters: "wrongful dismissal" should be compared against the
    concept vocabulary as a phrase before its individual words are, so the
    two-word concept wins over an accidental single-word near-match.
    """
    tokens = query.split()
    if not tokens:
        return []

    grams: list[str] = []
    for size in range(min(max_n, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            gram = " ".join(tokens[start : start + size])
            if len(gram) >= MIN_FUZZY_CHARS:
                grams.append(gram)
    return grams


def ground_query(text: str) -> list[str]:
    """
    Return canonical Concept.name values matched in `text`.

    Strategy:
      1. Substring match: any concept name/alias found inside the
         lowercased, whitespace-normalized query grounds that concept.
         Multiple concepts can match one query.
      2. If nothing matches, fall back to fuzzy matching over the query's word
         N-GRAMS (difflib), above FUZZY_CUTOFF.

    Returns an empty list if nothing matches either way.

    WHY N-GRAMS. The fallback used to compare the WHOLE query string against
    each concept term:

        get_close_matches("my employer has not paid me since january",
                          ["salary delay", "gratuity", ...], cutoff=0.8)

    A sentence and a two-word concept never reach a 0.8 similarity ratio no
    matter how well they correspond, so the fallback was effectively dead code
    for real user input — it could only ever fire on a query that was already
    almost exactly a concept name. Scoring n-grams instead compares like with
    like, so a misspelling ("gratuty", "retrenchement") or a slight variant
    ("wrongful dismissal" vs "wrongful termination") now grounds as intended.
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

    # Stage 2 — paraphrase matching by token overlap. This is what catches the
    # extraction LLM's own wording, which is rarely a verbatim alias.
    #
    # Deliberately UNIONED with stage 1 rather than used only as a fallback. A
    # substring hit is not evidence that the substring hit is the BEST match:
    # "punishment for non-payment of wages" contains the alias "payment of
    # wages", so short-circuiting there grounded it to `salary delay` and the
    # user asking about penalties never saw the penalties provisions, even
    # though "punishment" matches that concept exactly. Both stages contribute;
    # graph/ranking.py decides what actually reaches the answer.
    for name in _overlap_matches(query)[:MAX_OVERLAP_MATCHES]:
        if name not in seen:
            seen.add(name)
            matched.append(name)

    if matched:
        return matched

    # Stage 3 — fuzzy fallback over n-grams, longest first. Stops at the first
    # n-gram that matches anything, so one typo grounds one concept rather than
    # dragging in every loosely similar term in the vocabulary.
    for gram in _ngrams(query):
        close = get_close_matches(
            gram, _ALL_TERMS.keys(), n=MAX_FUZZY_MATCHES, cutoff=FUZZY_CUTOFF
        )
        if not close:
            continue
        for term in close:
            name = _ALL_TERMS[term]
            if name not in seen:
                seen.add(name)
                matched.append(name)
        if matched:
            return matched

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