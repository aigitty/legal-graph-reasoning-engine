"""
graph/ranking.py

Deterministic relevance ranking for retrieved sections.

THE PROBLEM THIS SOLVES
-----------------------
Traversal used to return up to MAX_SECTIONS sections ordered by
(act_id, section_number) — i.e. arbitrarily — and the cap sliced that arbitrary
order. Two consequences:

  * the evidence pack led with whatever happened to sort first, so the synthesis
    LLM saw the sections that answered the question buried behind ones that did
    not, and
  * trimming the pack was unsafe. config.CITIZEN_EVIDENCE_PACK_LIMIT was
    disabled precisely because a cap of 5 dropped the sections that actually
    answered a deductions query.

Ranking makes the ORDER meaningful, so the cap keeps the best evidence instead
of the alphabetically luckiest.

NO LLM. NO EMBEDDINGS. NO NETWORK. Every signal below is computed from data the
graph already returned, so ranking stays as auditable as the traversal that fed
it — which is the whole point of the project.

THE SIGNALS
-----------
  relevance      APPLIES_TO said 'primary' vs 'supporting' — the curated
                 ontology judgement, and the strongest single signal.
  concept_hits   how many of the user's grounded concepts this section anchors.
                 A section reached from two different concepts in one query is
                 usually the section the question sits on.
  hop_distance   0 for an anchor, 1..n for a section reached by CITES expansion.
                 Decays: a section three hops out is context, not the answer.
  lexical        Okapi BM25 of the query against the section title + text, with
                 the title weighted up. IDF is computed over the CANDIDATE SET
                 (not the whole corpus) because that is what discriminates
                 between the sections actually competing for a slot.
  act_priority   from act_metadata.json — prefers the operative consolidating
                 Code over older overlapping law when both are in force.

Weights live in config.py (RANK_W_*) and are normalised, so tuning them never
silently changes the score range.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from config import cfg

# Small, fixed English + legal-boilerplate stoplist. Deliberately not learned:
# a fixed list keeps ranking reproducible across runs and machines.
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from by with without under over
    is are was were be been being do does did doing have has had having
    i me my we our you your he him his she her it its they them their this that
    these those what which who whom when where why how
    shall may can will would should could must not no nor so than then there here
    as into about after before during any each other such same own too very
    section sections act code sub clause provided provision provisions
    please tell explain want know need get got question
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 parameters — standard defaults; k1 controls term-frequency saturation,
# b controls length normalisation.
_BM25_K1 = 1.5
_BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords and 1-char noise removed."""
    return [
        token
        for token in _TOKEN_RE.findall(str(text).lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


@dataclass
class RankCandidate:
    """
    One section competing for a slot in the evidence pack.

    `raw` is carried through untouched so the caller can recover whatever object
    it started with (a dict from the graph layer, or a SectionContext).
    """

    section_id: str
    title: str
    text: str
    relevance: str = "supporting"
    hop_distance: int = 0
    concept_hits: int = 1
    act_priority: int = 0
    raw: object = None

    # populated by rank()
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bm25_scores(
    query_tokens: Sequence[str],
    documents: Sequence[Sequence[str]],
) -> list[float]:
    """
    Okapi BM25 of one query against a candidate set, normalised to [0, 1].

    IDF is computed over `documents` only — the sections competing in THIS query
    — so a term appearing in every candidate contributes nothing, while a term
    appearing in one candidate is highly discriminative. Returns all-zeros when
    the query has no usable tokens or the candidates are empty.
    """
    if not query_tokens or not documents:
        return [0.0] * len(documents)

    doc_count = len(documents)
    lengths = [len(doc) for doc in documents]
    avg_length = (sum(lengths) / doc_count) or 1.0

    doc_frequency: dict[str, int] = {}
    doc_term_counts: list[dict[str, int]] = []
    for doc in documents:
        counts: dict[str, int] = {}
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
        doc_term_counts.append(counts)
        for token in counts:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    raw_scores: list[float] = []
    for index, counts in enumerate(doc_term_counts):
        score = 0.0
        length_norm = _BM25_K1 * (
            1 - _BM25_B + _BM25_B * (lengths[index] / avg_length)
        )
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            # +1 inside the log keeps IDF non-negative for terms present in
            # every document (standard BM25+ style smoothing).
            idf = math.log(
                1 + (doc_count - doc_frequency[token] + 0.5) / (doc_frequency[token] + 0.5)
            )
            score += idf * (frequency * (_BM25_K1 + 1)) / (frequency + length_norm)
        raw_scores.append(score)

    peak = max(raw_scores)
    if peak <= 0:
        return [0.0] * doc_count
    return [score / peak for score in raw_scores]


def _relevance_score(relevance: str) -> float:
    return 1.0 if str(relevance).lower() == "primary" else cfg.RANK_SUPPORTING_FLOOR


def _hop_score(hop_distance: int) -> float:
    # 1.0 at the anchor, decaying with distance: 1/(1+d).
    return 1.0 / (1.0 + max(0, int(hop_distance)))


def rank(
    query: str,
    candidates: Iterable[RankCandidate],
    title_weight: int | None = None,
) -> list[RankCandidate]:
    """
    Score and sort candidates, best first. Returns the same objects with `score`
    and `score_parts` populated, so a caller can show the breakdown.

    Ties break on (relevance, hop_distance, section_id) to keep the order
    STABLE — two runs of the same query must produce the same evidence pack, or
    the whole pipeline stops being reproducible.
    """
    items = list(candidates)
    if not items:
        return []

    repeat = cfg.RANK_TITLE_WEIGHT if title_weight is None else title_weight
    query_tokens = tokenize(query)
    documents = [
        tokenize(item.title) * max(1, repeat) + tokenize(item.text)
        for item in items
    ]
    lexical_scores = _bm25_scores(query_tokens, documents)

    max_hits = max((item.concept_hits for item in items), default=1) or 1
    max_priority = max((item.act_priority for item in items), default=1) or 1

    total_weight = (
        cfg.RANK_W_RELEVANCE
        + cfg.RANK_W_CONCEPT_HITS
        + cfg.RANK_W_HOP
        + cfg.RANK_W_LEXICAL
        + cfg.RANK_W_ACT_PRIORITY
    ) or 1.0

    for item, lexical in zip(items, lexical_scores):
        parts = {
            "relevance": round(_relevance_score(item.relevance), 4),
            "concept_hits": round(_clamp(item.concept_hits / max_hits), 4),
            "hop": round(_hop_score(item.hop_distance), 4),
            "lexical": round(_clamp(lexical), 4),
            "act_priority": round(_clamp(item.act_priority / max_priority), 4),
        }
        item.score_parts = parts
        item.score = round(
            (
                cfg.RANK_W_RELEVANCE * parts["relevance"]
                + cfg.RANK_W_CONCEPT_HITS * parts["concept_hits"]
                + cfg.RANK_W_HOP * parts["hop"]
                + cfg.RANK_W_LEXICAL * parts["lexical"]
                + cfg.RANK_W_ACT_PRIORITY * parts["act_priority"]
            )
            / total_weight,
            6,
        )

    items.sort(
        key=lambda c: (
            -c.score,
            0 if str(c.relevance).lower() == "primary" else 1,
            c.hop_distance,
            c.section_id,
        )
    )
    return items


def rank_and_cap(
    query: str,
    candidates: Iterable[RankCandidate],
    limit: int,
    keep_all_primaries: bool = True,
) -> list[RankCandidate]:
    """
    Rank, then cut to `limit`.

    `keep_all_primaries` guarantees that a curated PRIMARY anchor is never
    dropped in favour of a better-scoring SUPPORTING section. The ontology
    asserted that primaries are the operative provisions for the concept; the
    cap exists to control context size, not to overrule that judgement. This is
    what makes capping safe enough to re-enable for the citizen persona.
    """
    ranked = rank(query, candidates)
    if limit <= 0 or len(ranked) <= limit:
        return ranked

    if not keep_all_primaries:
        return ranked[:limit]

    primaries = [c for c in ranked if str(c.relevance).lower() == "primary"]
    others = [c for c in ranked if str(c.relevance).lower() != "primary"]

    kept = primaries[:limit]
    remaining = limit - len(kept)
    if remaining > 0:
        kept += others[:remaining]

    # Restore global ranked order among the survivors.
    kept_ids = {c.section_id for c in kept}
    return [c for c in ranked if c.section_id in kept_ids]
