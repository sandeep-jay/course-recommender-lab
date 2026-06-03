"""Contract tests for the semantic-vector recommenders (Phase 3).

The SBERT backend must honor the :class:`Recommender` contract: exclude the seed,
return ``list[Rec]`` of length ``<= k`` sorted by descending score, survive
sparse-text courses, serve free-text queries, and persist/reload an artifact.
The API backend must *degrade gracefully* — skip (raise
:class:`EmbeddingsUnavailable`) when no key is set, never hard-fail.

To stay fast, the whole module shares one tmp artifact + embedding-cache dir and
one fitted SBERT instance: the model (MiniLM) is loaded and the five docs encoded
exactly once; every later fit is a cache hit. ``sentence-transformers`` is an
optional extra, so the SBERT tests skip cleanly when it is not installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.data import _build_text
from courserec.interfaces import Rec
from courserec.recommenders import embeddings as E
from courserec.recommenders.embeddings import (
    ApiEmbeddingRecommender,
    EmbeddingsUnavailable,
    SbertRecommender,
)

_MODEL = "all-MiniLM-L6-v2"

# Skip the local-encoder tests when the heavy 'semantic' extra is absent.
_HAS_SBERT = True
try:  # pragma: no cover - import guard
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_SBERT = False

requires_sbert = pytest.mark.skipif(
    not _HAS_SBERT, reason="install the 'semantic' extra (sentence-transformers, torch)"
)


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    """A tiny catalog with twins, distinct topics, and a sparse-text course."""
    rows = [
        (
            "AEROENG C124",
            "AEROENG",
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation.",
        ),
        (
            "MATSCI C135",
            "MATSCI",
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation loads.",
        ),
        (
            "AEROENG 1",
            "AEROENG",
            "Introduction to Flight",
            "Aerodynamics, lift, drag, and the principles of flight.",
        ),
        (
            "MUSIC 10",
            "MUSIC",
            "Introduction to Music Theory",
            "Harmony, counterpoint, melody, and rhythm.",
        ),
        ("STAT 20", "STAT", "Introduction to Statistics", None),  # sparse: title only
    ]
    df = pd.DataFrame(rows, columns=["course_id", "subject", "title", "description"])
    df["text"] = [
        _build_text(t, d) for t, d in zip(df["title"], df["description"], strict=False)
    ]
    return df.set_index("course_id")


@pytest.fixture(scope="module", autouse=True)
def _isolate_artifacts(tmp_path_factory):
    """Redirect both cache layers to a shared module-tmp dir (restore afterward)."""
    root = tmp_path_factory.mktemp("emb_artifacts")
    saved = (E.ARTIFACTS_DIR, E._EMBCACHE_DIR)
    E.ARTIFACTS_DIR = root
    E._EMBCACHE_DIR = root / "embcache"
    yield
    E.ARTIFACTS_DIR, E._EMBCACHE_DIR = saved


@pytest.fixture(scope="module")
def rec(catalog: pd.DataFrame) -> SbertRecommender:
    """One SBERT recommender fitted once for the whole module (model load amortized)."""
    r = SbertRecommender(model_name=_MODEL)
    r.fit(catalog)
    return r


@requires_sbert
def test_recommend_similar_excludes_seed(rec: SbertRecommender) -> None:
    """recommend_similar never returns the seed itself."""
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert all(isinstance(r, Rec) for r in recs)
    assert "AEROENG C124" not in {r.course_id for r in recs}


@requires_sbert
def test_sorted_and_capped(rec: SbertRecommender) -> None:
    """Results are sorted by descending score and never exceed k."""
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert len(recs) <= 3
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


@requires_sbert
def test_twin_ranks_first(rec: SbertRecommender) -> None:
    """The near-identical cross-listed twin is the top recommendation."""
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert recs[0].course_id == "MATSCI C135"


@requires_sbert
def test_sparse_text_course_does_not_crash(rec: SbertRecommender) -> None:
    """A title-only course is embedded and served, not crashed on."""
    recs = rec.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


@requires_sbert
def test_recommend_by_text_is_semantic(rec: SbertRecommender) -> None:
    """A free-text query matches on meaning, not shared words.

    'flight aerodynamics and lift' shares no full title with 'Introduction to
    Flight', yet the encoder ranks that course first — the semantic payoff.
    """
    recs = rec.recommend_by_text("flight aerodynamics and lift", k=3)
    assert recs
    assert recs[0].course_id == "AEROENG 1"
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


@requires_sbert
def test_empty_query_returns_empty(rec: SbertRecommender) -> None:
    """A query that normalizes to nothing yields an empty list, not garbage."""
    assert rec.recommend_by_text("   ", k=3) == []


@requires_sbert
def test_unknown_seed_raises(rec: SbertRecommender) -> None:
    """An unknown seed id raises a clear KeyError."""
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


@requires_sbert
def test_artifact_cache_roundtrips(
    catalog: pd.DataFrame, rec: SbertRecommender
) -> None:
    """A second instance with the same config reloads from the persisted artifact."""
    assert rec._artifact_dir.joinpath("meta.json").exists()
    reloaded = SbertRecommender(model_name=_MODEL)
    reloaded.fit(catalog)  # fingerprint matches -> loads, no re-encode
    a = [r.course_id for r in rec.recommend_similar("AEROENG C124", k=3)]
    b = [r.course_id for r in reloaded.recommend_similar("AEROENG C124", k=3)]
    assert a == b


def test_invalid_index_type_rejected() -> None:
    """An unsupported index_type is rejected at construction."""
    with pytest.raises(ValueError):
        SbertRecommender(model_name=_MODEL, index_type="bogus")


def test_api_backend_skips_without_key(monkeypatch, catalog: pd.DataFrame) -> None:
    """The API backend degrades gracefully (skips) when no key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rec = ApiEmbeddingRecommender()
    with pytest.raises(EmbeddingsUnavailable):
        rec.fit(catalog)
