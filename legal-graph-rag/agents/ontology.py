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
    happen happens happened let lost
    """.split()
)
# The last line above is not filler — those words are the ENTIRE content of a
# real alias once the rest is stopworded, so they became single-token overlap
# matches that fire on ordinary English. "what happens to the employer" (an
# alias of `penalties for employer offences`) reduces to "happen", which is rare
# across the vocabulary and therefore passes the distinctiveness guard with a
# perfect score — so "my company was taken over, what happens to my service?"
# grounded to employer penalties and pulled six penalty and inspection sections
# into the pack as PRIMARIES, crowding out the transfer and gratuity provisions
# the question was actually about. Stopwording them costs nothing: stage-1
# substring matching is not stopworded, so "what happens to the employer"
# written verbatim still grounds exactly as before.

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


# Derivational endings stripped so a query's VERB meets the vocabulary's NOUN.
# Longest first, so "retrenchment" loses "ment" rather than being left alone.
_SUFFIXES = ("ment", "ing", "ed")

# A stem shorter than this is not a word, it is a collision waiting to happen:
# "need" -> "ne", "used" -> "us", "payment" -> "pay". Keeping the original in
# those cases costs a rare match; stripping them would ground on noise.
_MIN_STEM = 4


def _norm_token(token: str) -> str:
    """
    Reduce a token to a comparable stem.

    Handles plurals ('wages'/'wage', 'deductions'/'deduction') AND the
    verb/noun split, which was a live grounding hole: the concept vocabulary is
    written in nouns ("retrenchment", "dismissal") while users write verbs
    ("must an employer RETRENCH workmen?"). "retrench" is not a substring of
    "retrenchment" and shares no token with it, so that query grounded to
    `compensation for injury at work` — matched on the stray word "workmen" —
    and missed retrenchment entirely. Anyone whose phrasing the extraction LLM
    did not happen to rewrite into the noun form got injury-compensation law
    for a termination question.
    """
    if len(token) > 3 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
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

# Token set of each concept's CANONICAL NAME only (no aliases). Used to exempt
# single-word concepts from the distinctiveness guard below — see
# _overlap_matches.
_NAME_TOKENS: dict[str, frozenset[str]] = {
    concept.name: frozenset(_content_tokens(concept.name)) for concept in CONCEPTS
}

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
        #
        # EXEMPTION: the shared token covers the concept's whole CANONICAL NAME.
        # For a single-word concept — `retrenchment`, `gratuity`, `bonus` — the
        # user has written the concept itself, which is the strongest possible
        # signal, yet document frequency counts it as common precisely BECAUSE
        # it is the topic (four concepts mention retrenchment). That inverted the
        # guard: "must an employer retrench workmen?" was denied `retrenchment`
        # and kept only `compensation for injury at work`, matched on the stray
        # word "workmen".
        #
        # Deliberately keyed on the NAME, not on any alias. Exempting aliases too
        # would let "employ" (in the alias set of `re-employment of retrenched
        # workmen`, DF 9) ground every query containing the word "employment".
        if len(shared) == 1:
            token = next(iter(shared))
            covers_full_name = shared == _NAME_TOKENS.get(concept_name, frozenset())
            if (
                not covers_full_name
                and _TOKEN_DF.get(token, 0) > MAX_DF_FOR_SINGLE_TOKEN
            ):
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


# ---------------------------------------------------------------------------
# Companion concepts — the remedy layer
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Grounding answers "what is this question about". It does not answer "and what
# can this person actually DO about it", because a user never phrases that part:
# they write "in what order must an employer retrench workmen?", not "...and
# which forum hears a dispute about it".
#
# The consequence was visible in real output. The citizen prompt requires every
# "What you can do next" step to be grounded in a retrieved section and forbids
# inventing a forum, a deadline or an office — correctly, since an invented
# tribunal is worse than no answer. But retrieval never reached the dispute
# machinery, so there was nothing grounded to offer and the steps degraded to
# "ask your employer for the reasons" and "review your employment agreement".
# That is not a route to a remedy; it is what the reader could have written
# themselves.
#
# So: when a query grounds to a GRIEVANCE concept, deterministically also
# retrieve the concepts that carry the remedy for that family. No LLM, no
# guessing — a curated adjacency, validated against the vocabulary at import.
#
# Companion concepts contribute SUPPORTING sections only (see relevance_for):
# they are context for the answer, never the anchor of it. A retrenchment
# question must still be answered by the retrenchment sections.
_TERMINATION_REMEDIES = ("conciliation and settlement", "labour court and tribunal")
_WAGE_REMEDIES = ("recovering unpaid wages", "appeal against an order")
_BENEFIT_REMEDIES = ("recovering unpaid wages",)
_INSPECTION_REMEDIES = ("labour inspector",)

COMPANION_CONCEPTS: dict[str, tuple[str, ...]] = {
    # Job loss — the dispute machinery is the route.
    "wrongful termination": _TERMINATION_REMEDIES,
    "retrenchment": _TERMINATION_REMEDIES,
    "retrenchment compensation": _TERMINATION_REMEDIES,
    "re-employment of retrenched workmen": _TERMINATION_REMEDIES,
    "notice period": _TERMINATION_REMEDIES,
    "lay-off": _TERMINATION_REMEDIES,
    "factory closure": _TERMINATION_REMEDIES,
    "unfair labour practice": _TERMINATION_REMEDIES,
    "transfer of undertaking": _TERMINATION_REMEDIES,
    "conditions of service": _TERMINATION_REMEDIES,
    "standing orders": _TERMINATION_REMEDIES,
    # Money owed — the wage-claim authority is the route.
    "minimum wages": _WAGE_REMEDIES,
    "floor wage": _WAGE_REMEDIES,
    "overtime": _WAGE_REMEDIES,
    "salary delay": _WAGE_REMEDIES,
    "illegal wage deduction": _WAGE_REMEDIES,
    "deductions from wages": _WAGE_REMEDIES,
    "how and when wages must be paid": _WAGE_REMEDIES,
    "bonus": _WAGE_REMEDIES,
    "how bonus is calculated": _WAGE_REMEDIES,
    "gender discrimination in wages": _WAGE_REMEDIES,
    # Social-security benefits — recovery/appeal provisions of SSC 2020.
    "gratuity": _BENEFIT_REMEDIES,
    "gratuity nomination": _BENEFIT_REMEDIES,
    "provident fund": _BENEFIT_REMEDIES,
    "employees state insurance": _BENEFIT_REMEDIES,
    "maternity benefit": _BENEFIT_REMEDIES,
    "compensation for injury at work": _BENEFIT_REMEDIES,
    # Hours and leave — enforcement is by inspection, not a tribunal.
    "working hours": _INSPECTION_REMEDIES,
    "weekly holiday": _INSPECTION_REMEDIES,
    "annual leave": _INSPECTION_REMEDIES,
    "rest interval and spreadover": _INSPECTION_REMEDIES,
    "night work for women": _INSPECTION_REMEDIES,
    "child labour": _INSPECTION_REMEDIES,
}

# Fail at IMPORT if the table names a concept that does not exist. A silent typo
# here would disable a remedy for a whole family of queries and show up only as
# a vague answer months later — exactly the class of bug this table exists to
# fix.
_KNOWN_CONCEPT_NAMES = {concept.name for concept in CONCEPTS}
_BAD_COMPANIONS = sorted(
    {
        name
        for trigger, companions in COMPANION_CONCEPTS.items()
        for name in (trigger, *companions)
        if name not in _KNOWN_CONCEPT_NAMES
    }
)
if _BAD_COMPANIONS:
    raise ValueError(
        "COMPANION_CONCEPTS references concepts absent from concept_map.json: "
        + ", ".join(_BAD_COMPANIONS)
    )


def companion_concepts(grounded: list[str]) -> list[str]:
    """
    Remedy concepts implied by what the user actually asked about.

    Returns only concepts NOT already grounded — a user who explicitly asked
    about tribunals has them anchored as primaries already, and should not have
    them demoted to supporting by this layer.
    """
    already = set(grounded)
    extra: list[str] = []
    for concept_name in grounded:
        for companion in COMPANION_CONCEPTS.get(concept_name, ()):
            if companion not in already:
                already.add(companion)
                extra.append(companion)
    return extra


def relevance_for(
    concept_names: list[str],
    companion_names: list[str] | tuple[str, ...] = (),
) -> dict[str, str]:
    """
    Merge section relevance across the given concept names.

    "primary" wins over "supporting" if a section appears under multiple
    concepts with different relevance.

    Sections reached ONLY through a companion concept are always "supporting",
    however they are curated. A tribunal-constitution provision is genuinely
    primary for the question "which court hears my dispute", but it must not
    outrank the retrenchment sections for someone who asked about retrenchment
    and never mentioned a court.
    """
    merged: dict[str, str] = {}
    for concept in CONCEPTS:
        if concept.name not in concept_names:
            continue
        for section_id, relevance in concept.section_relevance.items():
            if merged.get(section_id) != "primary":
                merged[section_id] = relevance

    for concept in CONCEPTS:
        if concept.name not in companion_names:
            continue
        for section_id in concept.section_relevance:
            merged.setdefault(section_id, "supporting")
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