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

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from courserec.config import RANDOM_SEED
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


def build_crosslist_truth(courses: pd.DataFrame) -> CrossListTruth:
    """Build the cross-listing ground truth: seed -> set of twin ``course_id``s.

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

    per_query = {f"{m}@{k}": [] for m in ("recall", "precision", "ndcg") for k in ks}
    aps, rrs, same_subj, diversities, all_recs = [], [], [], [], []

    start = time.perf_counter()
    for seed in seeds:
        ranked = [r.course_id for r in rec.recommend_similar(seed, k=max_k)]
        all_recs.append(ranked)
        relevant = truth[seed]
        for k in ks:
            per_query[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            per_query[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            per_query[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
        aps.append(average_precision(ranked, relevant))
        rrs.append(reciprocal_rank(ranked, relevant))
        seed_subj = subjects.get(seed)
        top10 = ranked[:10]
        same_subj.append(
            np.mean([subjects.get(c) == seed_subj for c in top10]) if top10 else 0.0
        )
        diversities.append(intra_list_diversity(top10, ref_matrix, ref_row))
    elapsed = time.perf_counter() - start

    metrics = {name: float(np.mean(vals)) for name, vals in per_query.items()}
    metrics["map"] = float(np.mean(aps))
    metrics["mrr"] = float(np.mean(rrs))

    ndcg10 = np.array(per_query["ndcg@10"])
    return EvalResult(
        name=rec.name,
        config=rec.config,
        n_queries=len(seeds),
        metrics=metrics,
        ndcg10_ci=bootstrap_ci(ndcg10, n_boot=n_boot),
        same_subject_at_10=float(np.mean(same_subj)) if same_subj else 0.0,
        coverage=coverage(all_recs, len(courses)),
        diversity=float(np.mean(diversities)) if diversities else 0.0,
        novelty=novelty(all_recs),
        fit_time_s=fit_time_s,
        query_latency_ms=1000.0 * elapsed / len(seeds) if seeds else 0.0,
    )
