"""Contract tests for the topic recommenders (Phase 2): LSA, NMF, LDA.

Each technique must honor the :class:`Recommender` contract: exclude the seed,
return ``list[Rec]`` of length ``<= k`` sorted by descending score, survive
sparse-text courses, and persist/reload an artifact. Plus a topic-specific
interpretation check (the topic→term table is populated after fit). These run
against the tiny synthetic catalog for speed, with a small ``n_topics`` so the
factorizations are well-posed on five documents.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.interfaces import Rec
from courserec.recommenders import topics
from courserec.recommenders.topics import (
    LDARecommender,
    LSARecommender,
    NMFRecommender,
)

# Small topic count: the mini catalog has only 5 documents, so the reduced
# space must stay low-rank to be well-defined for all three factorizations.
_N_TOPICS = 3

RECOMMENDERS = [LSARecommender, NMFRecommender, LDARecommender]


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Redirect artifact persistence to a temp dir so tests never touch the repo."""
    monkeypatch.setattr(topics, "ARTIFACTS_DIR", tmp_path / "artifacts")


def _make(cls):
    """Instantiate each technique under a small representative config."""
    return cls(n_topics=_N_TOPICS, ngram_max=1, title_weight=2)


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
def test_twin_ranks_near_top(cls, mini_catalog: pd.DataFrame) -> None:
    """The cross-listed twin (near-identical text) ranks among the top results.

    Topic reduction can blur the near-duplicate signal lexical methods nail, so
    we assert "top 2" rather than "top 1" — a meaningful quality smoke without
    being brittle on a five-document corpus.
    """
    rec = _make(cls)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert "MATSCI C135" in {r.course_id for r in recs[:2]}


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
def test_out_of_vocab_query_returns_empty(cls, mini_catalog: pd.DataFrame) -> None:
    """A query with no in-vocabulary terms yields an empty list, not garbage."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    assert rec.recommend_by_text("zzzz qqqq xxxx", k=3) == []


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_unknown_seed_raises(cls, mini_catalog: pd.DataFrame) -> None:
    """An unknown seed id raises a clear KeyError rather than returning garbage."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_topic_terms_populated(cls, mini_catalog: pd.DataFrame) -> None:
    """The interpretation table is built: each topic has non-empty top terms."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    for topic in range(_N_TOPICS):
        terms = rec.topic_terms(topic, n=5)
        assert isinstance(terms, list)
        assert terms  # at least one term per topic


@pytest.mark.parametrize("cls", RECOMMENDERS)
def test_artifact_cache_roundtrips(cls, mini_catalog: pd.DataFrame) -> None:
    """A second instance with the same config loads from the persisted artifact."""
    rec = _make(cls)
    rec.fit(mini_catalog)
    assert rec._artifact_dir.joinpath("meta.json").exists()

    reloaded = _make(cls)
    from courserec.recommenders.topics import _build_docs

    docs = _build_docs(mini_catalog, reloaded.config["title_weight"])
    fingerprint = reloaded._fingerprint(docs)
    reloaded._course_ids = list(mini_catalog.index)
    reloaded._row = {c: i for i, c in enumerate(reloaded._course_ids)}
    assert reloaded._load(fingerprint) is True

    a = rec.recommend_similar("AEROENG C124", k=3)
    b = reloaded.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in a] == [r.course_id for r in b]


def test_distinct_classes_do_not_share_artifacts(mini_catalog: pd.DataFrame) -> None:
    """LSA and NMF under the same config must not load each other's artifact.

    Their fingerprints include the class name, so a same-config sibling is a
    cache miss rather than a silently wrong load.
    """
    from courserec.recommenders.topics import _build_docs

    lsa = LSARecommender(n_topics=_N_TOPICS, ngram_max=1, title_weight=2)
    lsa.fit(mini_catalog)

    nmf = NMFRecommender(n_topics=_N_TOPICS, ngram_max=1, title_weight=2)
    nmf._course_ids = list(mini_catalog.index)
    nmf._row = {c: i for i, c in enumerate(nmf._course_ids)}
    docs = _build_docs(mini_catalog, nmf.config["title_weight"])
    assert nmf._load(nmf._fingerprint(docs)) is False


def test_invalid_config_rejected() -> None:
    """Out-of-range hyperparameters raise ValueError at construction."""
    with pytest.raises(ValueError):
        LSARecommender(n_topics=1)
    with pytest.raises(ValueError):
        NMFRecommender(n_topics=_N_TOPICS, ngram_max=0)
    with pytest.raises(ValueError):
        LDARecommender(n_topics=_N_TOPICS, title_weight=0)
