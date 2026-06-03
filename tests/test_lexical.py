"""Contract tests for the lexical recommenders (Phase 1).

Every technique must honor the :class:`Recommender` contract: exclude the seed,
return ``list[Rec]`` of length ``<= k`` sorted by descending score, and survive
sparse-text courses. These run against a tiny synthetic catalog for speed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.interfaces import Rec
from courserec.recommenders import lexical
from courserec.recommenders.lexical import (
    BM25Recommender,
    TfidfRecommender,
    bm25_weight_matrix,
)


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Redirect artifact persistence to a temp dir so tests never touch the repo."""
    monkeypatch.setattr(lexical, "ARTIFACTS_DIR", tmp_path / "artifacts")


def _make(cls):
    """Instantiate each technique under a small representative config."""
    return cls(ngram_max=2, title_weight=2)


RECOMMENDERS = [TfidfRecommender, BM25Recommender]


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_recommend_similar_excludes_seed(cls, mini_catalog: pd.DataFrame) -> None:
    """recommend_similar never returns the seed itself."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert all(isinstance(r, Rec) for r in recs)
    assert "AEROENG C124" not in {r.course_id for r in recs}


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_recommend_similar_sorted_and_capped(cls, mini_catalog: pd.DataFrame) -> None:
    """Results are sorted by descending score and never exceed k."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert len(recs) <= 3
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_twin_ranks_first(cls, mini_catalog: pd.DataFrame) -> None:
    """The cross-listed twin (near-identical text) is the top recommendation."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert recs[0].course_id == "MATSCI C135"


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_sparse_text_course_does_not_crash(cls, mini_catalog: pd.DataFrame) -> None:
    """A course with no description (title-only) is handled, not crashed on."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_recommend_by_text(cls, mini_catalog: pd.DataFrame) -> None:
    """Free-text query returns sorted, capped recommendations."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_by_text("composite materials heat", k=2)
    assert len(recs) <= 2
    assert all(isinstance(r, Rec) for r in recs)
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_unknown_seed_raises(cls, mini_catalog: pd.DataFrame) -> None:
    """An unknown seed id raises a clear KeyError rather than returning garbage."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_artifact_cache_roundtrips(cls, mini_catalog: pd.DataFrame) -> None:
    """A second instance with the same config loads from the persisted artifact."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    assert rec._artifact_dir.joinpath("meta.json").exists()
    reloaded = _make(cls)
    fingerprint = reloaded._fingerprint(reloaded._build_docs(mini_catalog))
    reloaded._course_ids = list(mini_catalog.index)
    reloaded._row = {c: i for i, c in enumerate(reloaded._course_ids)}
    assert reloaded._load(fingerprint) is True
    a = rec.recommend_similar("AEROENG C124", k=3)
    b = reloaded.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in a] == [r.course_id for r in b]


def test_bm25_weight_matrix_shape_and_sign(mini_catalog: pd.DataFrame) -> None:
    """The BM25 weight helper preserves sparsity shape and is non-negative."""
    from sklearn.feature_extraction.text import CountVectorizer

    counts = CountVectorizer().fit_transform(mini_catalog["text"]).tocsr()
    weights = bm25_weight_matrix(counts, k1=1.5, b=0.75)
    assert weights.shape == counts.shape
    assert weights.nnz == counts.nnz
    assert (weights.data >= 0).all()
