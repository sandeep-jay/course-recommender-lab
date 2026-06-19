"""The shared evaluation harness: ground truth, metrics, and lenses.

Because the catalog has no clicks or ratings, ground truth is constructed rather
than observed (recommender_plan.md §3). Two automatic lenses run here:

1. **Cross-listing pairs (primary).** ~10% of courses declare a cross-listed
   twin; a good item-to-item ranker puts the twin near the top. This validates
   correctness more than quality — twins share near-identical text, so lexical
   methods nail it — but it is the only fully automatic relevance signal.
2. **Same-subject coherence (sanity floor).** Fraction of the top-k sharing the
   seed's subject. Reported, never optimized: a subject-only model would max it
   while being useless.

The ground-truth set is small, so the primary metric (NDCG@10) is reported with
a bootstrap confidence interval — never crown a winner on a sub-CI gap.

Leakage note: ``Cross-Listed Course(s)`` is the target here, so no scored model
may read it as a feature (enforced by the recommenders, not this module).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from courserec.config import JUDGED_QUERIES_JSON, RANDOM_SEED
from courserec.interfaces import Recommender

logger = logging.getLogger(__name__)

CrossListTruth = dict[str, set[str]]


# --------------------------------------------------------------------------- #
# Ground truth                                                                #
# --------------------------------------------------------------------------- #


def _resolve_refs(value: str, nospace: dict[str, str]) -> list[str]:
    """Resolve a raw cross-listed cell to catalog ``course_id``s.

    The catalog stores cross-listings as ``"{SUBJECT}{NUMBER} {truncated title}"``
    (subject and number space-stripped), comma-separated for multiples. We match
    the leading space-stripped token against a lookup of space-stripped catalog
    ids and silently drop references to courses outside the catalog.

    Args:
        value: Raw ``cross_listed`` cell.
        nospace: Map from space-stripped ``course_id`` to ``course_id``.

    Returns:
        The resolved catalog ``course_id``s (possibly empty).
    """
    out: list[str] = []
    for part in str(value).split(","):
        token = part.strip().split(" ", 1)[0]
        if token in nospace:
            out.append(nospace[token])
    return out


def crosslist_edges(courses: pd.DataFrame) -> set[frozenset[str]]:
    """Resolve every cross-listing into an undirected ``{a, b}`` course-id edge.

    The single source of truth for which courses are cross-listed: both the
    ground-truth builder and the graph technique derive their edges from here so
    a held-out split and the graph's training set stay perfectly consistent.

    Args:
        courses: Processed catalog indexed by ``course_id`` with a
            ``cross_listed`` column.

    Returns:
        The set of undirected edges (2-element frozensets); self-references and
        out-of-catalog references are dropped.
    """
    nospace = {cid.replace(" ", ""): cid for cid in courses.index}
    edges: set[frozenset[str]] = set()
    for cid, value in courses["cross_listed"].dropna().items():
        for twin in _resolve_refs(value, nospace):
            if twin != cid:
                edges.add(frozenset((cid, twin)))
    return edges


def _edges_to_truth(edges: Iterable[frozenset[str]]) -> CrossListTruth:
    """Expand undirected edges into a symmetric seed -> twins truth mapping."""
    truth: CrossListTruth = {}
    for edge in edges:
        a, b = tuple(edge)
        truth.setdefault(a, set()).add(b)
        truth.setdefault(b, set()).add(a)
    return truth


def build_crosslist_truth(courses: pd.DataFrame) -> CrossListTruth:
    """Build the cross-listing ground truth: seed -> set of twin ``course_id``s.

    Kept directional (a seed is a course whose own ``cross_listed`` cell names an
    in-catalog twin) so the established cross-listing leaderboard is unchanged by
    Phase 5. The undirected, symmetric view used by the graph and its held-out
    split lives in :func:`crosslist_edges` instead.

    Args:
        courses: Processed catalog indexed by ``course_id`` with a
            ``cross_listed`` column.

    Returns:
        A mapping from each course that has at least one in-catalog cross-listed
        twin to the set of those twins (the seed itself is excluded).
    """
    nospace = {cid.replace(" ", ""): cid for cid in courses.index}
    truth: CrossListTruth = {}
    for cid, value in courses["cross_listed"].dropna().items():
        twins = {t for t in _resolve_refs(value, nospace) if t != cid}
        if twins:
            truth[cid] = twins
    logger.info("Cross-listing truth: %d seeds with in-catalog twins", len(truth))
    return truth


# --------------------------------------------------------------------------- #
# Held-out edge split (for the graph technique only — plan §2.6 leakage rule)  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CrossListSplit:
    """A leakage-free split of cross-listing edges for evaluating the graph model.

    The graph technique is the one method permitted to read ``Cross-Listed
    Course(s)`` as input, so it must be scored only on edges it never saw. This
    splits the undirected cross-listing edges into a training set (the graph may
    use these) and a held-out test set (the evaluation target).

    Attributes:
        train_truth: Seed -> twins from the training edges (what the graph sees).
        test_truth: Seed -> twins from the held-out edges (the eval target).
        held_out_edges: The undirected edges removed from training; the graph
            subtracts exactly these when building its adjacency.
    """

    train_truth: CrossListTruth
    test_truth: CrossListTruth
    held_out_edges: frozenset[frozenset[str]]


def split_crosslist_edges(
    truth: CrossListTruth, *, test_frac: float = 0.3, seed: int = RANDOM_SEED
) -> CrossListSplit:
    """Split cross-listing edges into train / held-out test sets reproducibly.

    Args:
        truth: Full cross-listing ground truth from :func:`build_crosslist_truth`.
        test_frac: Fraction of undirected edges to hold out for evaluation.
        seed: RNG seed (defaults to the global ``RANDOM_SEED``).

    Returns:
        A :class:`CrossListSplit`. A course may keep some training edges while a
        different edge of its own is held out — that is the realistic
        link-prediction setting we want.

    Raises:
        ValueError: If ``test_frac`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    # Sort edges to a canonical order so the permutation is deterministic across
    # runs regardless of set-iteration order.
    edges = sorted(
        ({frozenset((a, b)) for a, twins in truth.items() for b in twins}),
        key=lambda e: tuple(sorted(e)),
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    n_test = max(1, round(len(edges) * test_frac)) if edges else 0
    test_pos = set(perm[:n_test].tolist())

    train_edges = [e for i, e in enumerate(edges) if i not in test_pos]
    held_out = [e for i, e in enumerate(edges) if i in test_pos]
    logger.info(
        "Cross-listing edge split: %d train, %d held-out (test_frac=%.2f, seed=%d)",
        len(train_edges),
        len(held_out),
        test_frac,
        seed,
    )
    return CrossListSplit(
        train_truth=_edges_to_truth(train_edges),
        test_truth=_edges_to_truth(held_out),
        held_out_edges=frozenset(held_out),
    )


# --------------------------------------------------------------------------- #
# Ranking metrics (binary relevance)                                          #
# --------------------------------------------------------------------------- #


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items retrieved in the top ``k``."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` that are relevant."""
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """Average precision over the ranked list (the per-query term of MAP)."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant item (0 if none)."""
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary relevance."""
    if not relevant:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.array([1.0 if cid in relevant else 0.0 for cid in ranked[:k]])
    dcg = float((gains * discounts[: len(gains)]).sum())
    ideal = float(discounts[: min(len(relevant), k)].sum())
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------- #
# List-quality metrics (no ground truth)                                      #
# --------------------------------------------------------------------------- #


def build_reference_space(
    courses: pd.DataFrame,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """Build a technique-agnostic TF-IDF space for measuring intra-list diversity.

    Diversity must not be measured in the candidate model's own space (that would
    flatter it), so we fit one shared reference space over the combined text.

    Args:
        courses: Processed catalog indexed by ``course_id``.

    Returns:
        The L2-normalized TF-IDF matrix and a ``course_id`` -> row-index map.
    """
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(courses["text"].fillna("")).tocsr()
    row = {cid: i for i, cid in enumerate(courses.index)}
    return matrix, row


def intra_list_diversity(
    ranked: list[str], reference: sp.csr_matrix, row: dict[str, int]
) -> float:
    """Mean pairwise cosine *distance* among the recommended items (1 = diverse)."""
    rows = [row[c] for c in ranked if c in row]
    if len(rows) < 2:
        return 0.0
    sub = reference[rows]
    sims = (sub @ sub.T).toarray()
    iu = np.triu_indices(len(rows), k=1)
    return float(1.0 - sims[iu].mean())


def coverage(all_recs: list[list[str]], catalog_size: int) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    if catalog_size == 0:
        return 0.0
    seen = {cid for recs in all_recs for cid in recs}
    return len(seen) / catalog_size


def novelty(all_recs: list[list[str]]) -> float:
    """Mean self-information of recommended items by recommendation frequency.

    Items surfaced rarely across all seeds carry more information (are more
    novel). Defined as the mean over queries of the mean ``-log2(p)`` of their
    recommended items, where ``p`` is the item's share of all recommendations.
    """
    counts: dict[str, int] = {}
    total = 0
    for recs in all_recs:
        for cid in recs:
            counts[cid] = counts.get(cid, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    info = {cid: -np.log2(c / total) for cid, c in counts.items()}
    per_query = [float(np.mean([info[c] for c in recs])) for recs in all_recs if recs]
    return float(np.mean(per_query)) if per_query else 0.0


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Args:
        values: Per-query metric values.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level (0.05 -> 95% CI).

    Returns:
        ``(low, high)`` bounds of the CI on the mean.
    """
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


# --------------------------------------------------------------------------- #
# Result container + scorer                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class EvalResult:
    """All metrics for one technique×config, ready for a leaderboard row."""

    name: str
    config: dict
    n_queries: int
    metrics: dict[str, float]  # e.g. {"recall@5": .., "ndcg@10": ..}
    ndcg10_ci: tuple[float, float]
    same_subject_at_10: float
    coverage: float
    diversity: float
    novelty: float
    fit_time_s: float
    query_latency_ms: float
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        """Flatten to a single leaderboard row (one column per metric)."""
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            **{k: round(v, 4) for k, v in self.metrics.items()},
            "ndcg@10_ci_low": round(self.ndcg10_ci[0], 4),
            "ndcg@10_ci_high": round(self.ndcg10_ci[1], 4),
            "same_subject@10": round(self.same_subject_at_10, 4),
            "coverage": round(self.coverage, 4),
            "diversity": round(self.diversity, 4),
            "novelty": round(self.novelty, 4),
            "fit_time_s": round(self.fit_time_s, 3),
            "query_latency_ms": round(self.query_latency_ms, 3),
            "config": str(self.config),
        }


def _aggregate_ranking_metrics(
    rankings: Sequence[list[str]],
    relevants: Sequence[set[str]],
    ks: tuple[int, ...],
) -> tuple[dict[str, float], np.ndarray]:
    """Aggregate per-query ranking metrics shared by every lens.

    Both the cross-listing and the judged-text lenses score a list of ranked
    ``course_id``s against a per-query relevant set with the identical binary
    metrics; only the source of the rankings and the relevant sets differs. This
    helper owns that common reduction so the two scorers stay in lockstep.

    Args:
        rankings: One ranked ``course_id`` list per query.
        relevants: The relevant ``course_id`` set for each query (row-aligned
            with ``rankings``).
        ks: Cutoffs to report (the primary cutoff is 10).

    Returns:
        A ``(metrics, ndcg@10 per-query array)`` pair: the mean of every metric
        keyed ``"{metric}@{k}"`` plus ``"map"``/``"mrr"``, and the raw per-query
        NDCG@10 values for a bootstrap CI.
    """
    per_query = {f"{m}@{k}": [] for m in ("recall", "precision", "ndcg") for k in ks}
    aps, rrs = [], []
    for ranked, relevant in zip(rankings, relevants, strict=True):
        for k in ks:
            per_query[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            per_query[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            per_query[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
        aps.append(average_precision(ranked, relevant))
        rrs.append(reciprocal_rank(ranked, relevant))

    metrics = {
        name: float(np.mean(vals)) if vals else 0.0 for name, vals in per_query.items()
    }
    metrics["map"] = float(np.mean(aps)) if aps else 0.0
    metrics["mrr"] = float(np.mean(rrs)) if rrs else 0.0
    ndcg10 = np.array(per_query.get("ndcg@10", []), dtype=float)
    return metrics, ndcg10


def score_crosslist(
    rec: Recommender,
    courses: pd.DataFrame,
    truth: CrossListTruth,
    reference: tuple[sp.csr_matrix, dict[str, int]],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    fit_time_s: float = 0.0,
    n_boot: int = 1000,
) -> EvalResult:
    """Score a fitted recommender on the cross-listing + same-subject lenses.

    Args:
        rec: A recommender already ``fit`` on ``courses``.
        courses: Processed catalog indexed by ``course_id``.
        truth: Cross-listing ground truth from :func:`build_crosslist_truth`.
        reference: Output of :func:`build_reference_space` for diversity.
        ks: Cutoffs to report (the primary cutoff is 10).
        fit_time_s: Wall-clock fit time, passed through to the result row.
        n_boot: Bootstrap resamples for the NDCG@10 CI.

    Returns:
        An :class:`EvalResult` aggregating ranking and list-quality metrics.
    """
    ref_matrix, ref_row = reference
    max_k = max(ks)
    subjects = courses["subject"]
    seeds = list(truth)

    rankings, relevants, same_subj, diversities = [], [], [], []
    start = time.perf_counter()
    for seed in seeds:
        ranked = [r.course_id for r in rec.recommend_similar(seed, k=max_k)]
        rankings.append(ranked)
        relevants.append(truth[seed])
        seed_subj = subjects.get(seed)
        top10 = ranked[:10]
        same_subj.append(
            np.mean([subjects.get(c) == seed_subj for c in top10]) if top10 else 0.0
        )
        diversities.append(intra_list_diversity(top10, ref_matrix, ref_row))
    elapsed = time.perf_counter() - start

    metrics, ndcg10 = _aggregate_ranking_metrics(rankings, relevants, ks)
    return EvalResult(
        name=rec.name,
        config=rec.config,
        n_queries=len(seeds),
        metrics=metrics,
        ndcg10_ci=bootstrap_ci(ndcg10, n_boot=n_boot),
        same_subject_at_10=float(np.mean(same_subj)) if same_subj else 0.0,
        coverage=coverage(rankings, len(courses)),
        diversity=float(np.mean(diversities)) if diversities else 0.0,
        novelty=novelty(rankings),
        fit_time_s=fit_time_s,
        query_latency_ms=1000.0 * elapsed / len(seeds) if seeds else 0.0,
        extra={"lens": "crosslist"},
    )


# --------------------------------------------------------------------------- #
# Judged-text lens (recommend_by_text)                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgedQuery:
    """One hand-labeled free-text query and its relevant ``course_id`` set.

    Attributes:
        query: The natural-language query a user might type.
        relevant: The ``course_id``s a human judged clearly on-topic. The set is
            curated and necessarily incomplete (see :func:`load_judged_queries`),
            so it bounds *relative* recall across techniques, not absolute recall.
        note: A short rationale for the labels (documentation only, never scored).
    """

    query: str
    relevant: frozenset[str]
    note: str = ""


def load_judged_queries(
    path: Path = JUDGED_QUERIES_JSON,
    catalog_ids: Iterable[str] | None = None,
) -> list[JudgedQuery]:
    """Load the hand-labeled free-text ground truth, dropping stale references.

    This is the only ground truth for ``recommend_by_text`` (plan §3 lens 3). The
    catalog evolves between terms, so a label may reference a ``course_id`` that
    no longer exists; when ``catalog_ids`` is given, such labels are dropped with
    a warning (a memory recall reflecting an old file is not a reason to trust a
    stale id) and a query left with no in-catalog labels is skipped entirely.

    Args:
        path: Path to the judged-queries JSON (see ``data/judged_queries.json``).
        catalog_ids: If provided, the set of valid ``course_id``s; relevant
            labels outside it are dropped and fully-stale queries skipped.

    Returns:
        The judged queries with non-empty in-catalog relevant sets.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If an entry is missing the ``query`` or ``relevant`` field.
    """
    raw = json.loads(Path(path).read_text())
    known = set(catalog_ids) if catalog_ids is not None else None
    queries: list[JudgedQuery] = []
    for entry in raw["queries"]:
        relevant = set(entry["relevant"])
        if known is not None:
            dropped = relevant - known
            if dropped:
                logger.warning(
                    "judged query %r: dropping %d unknown course_id(s): %s",
                    entry["query"],
                    len(dropped),
                    sorted(dropped),
                )
            relevant &= known
        if not relevant:
            logger.warning(
                "judged query %r: no in-catalog relevant ids; skipping", entry["query"]
            )
            continue
        queries.append(
            JudgedQuery(entry["query"], frozenset(relevant), entry.get("note", ""))
        )
    logger.info(
        "Loaded %d judged queries (%d relevant labels) from %s",
        len(queries),
        sum(len(q.relevant) for q in queries),
        path,
    )
    return queries


def score_text_queries(
    rec: Recommender,
    queries: Sequence[JudgedQuery],
    courses: pd.DataFrame,
    reference: tuple[sp.csr_matrix, dict[str, int]],
    *,
    ks: tuple[int, ...] = (5, 10, 20),
    fit_time_s: float = 0.0,
    n_boot: int = 1000,
) -> EvalResult:
    """Score a recommender's ``recommend_by_text`` against the judged-query set.

    The free-text lens — the only one that measures the mode topic and semantic
    methods are meant to win. Same binary ranking metrics as the cross-listing
    lens, over the hand-labeled relevant sets. ``same_subject@10`` is undefined
    here (a free-text query has no seed subject) and is reported as NaN.

    Args:
        rec: A recommender already ``fit`` on ``courses``. Must implement
            ``recommend_by_text`` — text-incapable techniques are filtered out by
            the caller (see :func:`recommender_supports_text`), never passed here.
        queries: The judged queries from :func:`load_judged_queries`.
        courses: Processed catalog indexed by ``course_id`` (for list-quality
            metrics).
        reference: Output of :func:`build_reference_space` for diversity.
        ks: Cutoffs to report (the primary cutoff is 10).
        fit_time_s: Wall-clock fit time, passed through to the result row.
        n_boot: Bootstrap resamples for the NDCG@10 CI.

    Returns:
        An :class:`EvalResult` for the free-text lens (``extra["lens"] == "text"``).
    """
    ref_matrix, ref_row = reference
    max_k = max(ks)

    rankings, relevants, diversities = [], [], []
    start = time.perf_counter()
    for q in queries:
        ranked = [r.course_id for r in rec.recommend_by_text(q.query, k=max_k)]
        rankings.append(ranked)
        relevants.append(set(q.relevant))
        diversities.append(intra_list_diversity(ranked[:10], ref_matrix, ref_row))
    elapsed = time.perf_counter() - start

    metrics, ndcg10 = _aggregate_ranking_metrics(rankings, relevants, ks)
    return EvalResult(
        name=rec.name,
        config=rec.config,
        n_queries=len(queries),
        metrics=metrics,
        ndcg10_ci=bootstrap_ci(ndcg10, n_boot=n_boot),
        same_subject_at_10=float("nan"),  # undefined for free-text queries
        coverage=coverage(rankings, len(courses)),
        diversity=float(np.mean(diversities)) if diversities else 0.0,
        novelty=novelty(rankings),
        fit_time_s=fit_time_s,
        query_latency_ms=1000.0 * elapsed / len(queries) if queries else 0.0,
        extra={"lens": "text"},
    )


def recommender_supports_text(rec: Recommender) -> bool:
    """Return whether ``rec.recommend_by_text`` is implemented (not a stub).

    Item-to-item-only techniques may raise :class:`NotImplementedError` from
    ``recommend_by_text`` (interface contract). The free-text lens must skip
    those — and flag the gap — rather than crash the suite. We probe with a
    trivial query and treat only :class:`NotImplementedError` as "unsupported";
    any other behavior (including an empty result) counts as supported.

    Args:
        rec: A recommender already ``fit`` on the catalog.

    Returns:
        True if a probe query is served, False if it raises NotImplementedError.
    """
    try:
        rec.recommend_by_text("probe query", k=1)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 — any other error still means "implemented"
        return True
    return True
