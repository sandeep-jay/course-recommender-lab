"""Tests for the evaluation harness: truth resolution, metrics, and scoring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from courserec.eval import (
    JudgedQuery,
    average_precision,
    bootstrap_ci,
    build_crosslist_truth,
    build_reference_space,
    coverage,
    intra_list_diversity,
    load_judged_queries,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    recommender_supports_text,
    score_crosslist,
    score_text_queries,
)
from courserec.interfaces import Rec, Recommender
from courserec.recommenders.lexical import TfidfRecommender

# -- ground truth --------------------------------------------------------------


def test_crosslist_truth_resolves_twins(mini_catalog: pd.DataFrame) -> None:
    """Space-stripped references resolve to catalog ids; twins are mutual."""
    truth = build_crosslist_truth(mini_catalog)
    assert truth["AEROENG C124"] == {"MATSCI C135"}
    assert truth["MATSCI C135"] == {"AEROENG C124"}
    assert "AEROENG 1" not in truth  # no cross-listing


def test_crosslist_truth_excludes_self_and_out_of_catalog() -> None:
    """A reference to a non-catalog course is dropped, and self never appears."""
    df = pd.DataFrame(
        {
            "subject": ["X", "Y"],
            "cross_listed": ["GHOST999 NOT HERE", "XX1 X SELF REF"],
        },
        index=pd.Index(["X 1", "Y 1"], name="course_id"),
    )
    truth = build_crosslist_truth(df)
    assert truth == {}  # GHOST999 unresolved; XX1 != "X 1" token, so unresolved


# -- ranking metrics -----------------------------------------------------------


def test_ranking_metrics_known_values() -> None:
    """Hand-checked metrics on a small ranking with one relevant item at rank 2."""
    ranked = ["b", "a", "c"]
    rel = {"a"}
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1.0)
    assert recall_at_k(ranked, rel, 1) == pytest.approx(0.0)
    assert precision_at_k(ranked, rel, 2) == pytest.approx(0.5)
    assert reciprocal_rank(ranked, rel) == pytest.approx(0.5)
    assert average_precision(ranked, rel) == pytest.approx(0.5)
    # NDCG@2: gain at rank 2 = 1/log2(3); ideal gain at rank 1 = 1/log2(2) = 1.
    assert ndcg_at_k(ranked, rel, 2) == pytest.approx(1.0 / np.log2(3))


def test_metrics_handle_no_relevant_or_no_hits() -> None:
    """Empty relevant sets and miss-only rankings score zero, never crash."""
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0
    assert recall_at_k(["a"], {"z"}, 1) == 0.0
    assert reciprocal_rank(["a"], {"z"}) == 0.0
    assert average_precision(["a"], set()) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    """All relevant items at the top yields NDCG 1.0."""
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)


# -- list-quality metrics ------------------------------------------------------


def test_diversity_bounds(mini_catalog: pd.DataFrame) -> None:
    """Identical items -> 0 diversity; unrelated items -> positive diversity."""
    matrix, row = build_reference_space(mini_catalog)
    twins = intra_list_diversity(["AEROENG C124", "MATSCI C135"], matrix, row)
    mixed = intra_list_diversity(["AEROENG C124", "MUSIC 10"], matrix, row)
    assert 0.0 <= twins < mixed <= 1.0


def test_coverage_and_novelty() -> None:
    """Coverage is the fraction of catalog surfaced; novelty rewards rare items."""
    all_recs = [["a", "b"], ["a", "c"]]
    assert coverage(all_recs, catalog_size=4) == pytest.approx(3 / 4)
    # "a" appears in every list (low novelty); a list of only "a" is least novel.
    assert novelty([["a", "a"]]) == pytest.approx(0.0)
    assert novelty(all_recs) > 0.0


# -- bootstrap -----------------------------------------------------------------


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    """The CI brackets the sample mean and is reproducible under the global seed."""
    values = np.linspace(0.0, 1.0, 50)
    low, high = bootstrap_ci(values, n_boot=500)
    assert low <= values.mean() <= high
    assert (low, high) == bootstrap_ci(values, n_boot=500)


# -- end to end ----------------------------------------------------------------


def test_score_crosslist_end_to_end(mini_catalog: pd.DataFrame, monkeypatch) -> None:
    """A fitted recommender scores well when twins rank first, with a valid CI."""
    import tempfile
    from pathlib import Path

    from courserec.recommenders import lexical

    rec = TfidfRecommender(ngram_max=1)
    # Avoid touching the repo's artifacts dir during the test.
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(lexical, "ARTIFACTS_DIR", Path(tmp))
        rec.fit(mini_catalog)
        truth = build_crosslist_truth(mini_catalog)
        reference = build_reference_space(mini_catalog)
        result = score_crosslist(rec, mini_catalog, truth, reference, n_boot=100)
    assert result.n_queries == 2
    assert result.metrics["ndcg@10"] == pytest.approx(1.0)  # twins rank first
    assert (
        result.ndcg10_ci[0] <= result.metrics["ndcg@10"] <= result.ndcg10_ci[1] + 1e-9
    )
    assert 0.0 <= result.coverage <= 1.0


# -- judged free-text lens -----------------------------------------------------


def test_load_judged_queries_drops_stale_and_empty(tmp_path) -> None:
    """Unknown course_ids are dropped; a fully-stale query is skipped entirely."""
    path = tmp_path / "judged.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {"query": "machine learning", "relevant": ["A 1", "GHOST 9"]},
                    {"query": "all stale", "relevant": ["GHOST 1", "GHOST 2"]},
                ]
            }
        )
    )
    queries = load_judged_queries(path, catalog_ids={"A 1", "B 2"})
    assert len(queries) == 1  # the all-stale query is skipped
    assert queries[0].query == "machine learning"
    assert queries[0].relevant == frozenset({"A 1"})  # GHOST 9 dropped


def test_load_judged_queries_keeps_all_without_catalog(tmp_path) -> None:
    """Without a catalog filter, labels are kept verbatim (no resolution)."""
    path = tmp_path / "judged.json"
    path.write_text(
        json.dumps({"queries": [{"query": "q", "relevant": ["X 1"], "note": "n"}]})
    )
    queries = load_judged_queries(path)
    assert queries[0].relevant == frozenset({"X 1"})
    assert queries[0].note == "n"


class _FixedTextRecommender(Recommender):
    """A stub that returns a fixed ranking for any text query, to test the lens."""

    name = "fixed-text"
    config: dict = {}

    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    def fit(self, courses: pd.DataFrame) -> None:  # noqa: D102
        pass

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:  # noqa: D102
        raise NotImplementedError

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:  # noqa: D102
        return [Rec(c, 1.0 - i) for i, c in enumerate(self._ranking[:k])]


class _ItemOnlyRecommender(Recommender):
    """A stub that is item-to-item only (text mode unimplemented)."""

    name = "item-only"
    config: dict = {}

    def fit(self, courses: pd.DataFrame) -> None:  # noqa: D102
        pass

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:  # noqa: D102
        return []

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:  # noqa: D102
        raise NotImplementedError


def test_score_text_queries_metrics_and_nan_subject(mini_catalog: pd.DataFrame) -> None:
    """The text lens scores ranking metrics and reports same_subject@10 as NaN."""
    # One query whose single relevant item is returned at rank 1 -> perfect NDCG.
    queries = [JudgedQuery("composite materials", frozenset({"AEROENG C124"}))]
    rec = _FixedTextRecommender(["AEROENG C124", "MUSIC 10"])
    reference = build_reference_space(mini_catalog)
    result = score_text_queries(rec, queries, mini_catalog, reference, n_boot=50)
    assert result.n_queries == 1
    assert result.extra["lens"] == "text"
    assert result.metrics["ndcg@10"] == pytest.approx(1.0)
    assert np.isnan(result.same_subject_at_10)  # undefined for free-text
    assert 0.0 <= result.coverage <= 1.0


def test_recommender_supports_text() -> None:
    """Text capability is detected via a probe; only NotImplementedError = False."""
    assert recommender_supports_text(_FixedTextRecommender(["A 1"])) is True
    assert recommender_supports_text(_ItemOnlyRecommender()) is False
