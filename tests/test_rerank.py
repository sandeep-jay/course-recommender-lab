"""Contract tests for the retrieve→rerank→MMR recommender (Phase 4).

The pipeline must honor the :class:`Recommender` contract — exclude the seed,
return ``list[Rec]`` of length ``<= k`` sorted by descending score, survive
sparse text, serve free-text queries — and its defining behavior: lowering the
MMR ``λ`` must raise intra-list diversity (the phase-4 acceptance test,
recommender_plan.md §5).

To stay fast the module shares one tmp artifact/cache dir and reuses fitted
instances; the cross-encoder loads once. The whole module skips cleanly when the
optional ``semantic`` extra is absent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.data import _build_text
from courserec.eval import build_reference_space, intra_list_diversity
from courserec.interfaces import Rec
from courserec.recommenders import embeddings as E
from courserec.recommenders.rerank import RerankRecommender

# Skip the whole module when the heavy 'semantic' extra is absent.
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
    """A small catalog with twins, related courses, distinct topics, sparse text."""
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
            "AEROENG 2",
            "AEROENG",
            "Aircraft Performance",
            "Lift, drag, thrust, and the aerodynamics of flight vehicles.",
        ),
        (
            "MUSIC 10",
            "MUSIC",
            "Introduction to Music Theory",
            "Harmony, counterpoint, melody, and rhythm.",
        ),
        (
            "HISTORY 5",
            "HISTORY",
            "Modern European History",
            "Revolutions, nations, and war from 1789 to the present.",
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
    """Redirect both embedding-cache layers to a shared module-tmp dir."""
    root = tmp_path_factory.mktemp("rerank_artifacts")
    saved = (E.ARTIFACTS_DIR, E._EMBCACHE_DIR)
    E.ARTIFACTS_DIR = root
    E._EMBCACHE_DIR = root / "embcache"
    yield
    E.ARTIFACTS_DIR, E._EMBCACHE_DIR = saved


@pytest.fixture(scope="module")
def rec(catalog: pd.DataFrame) -> RerankRecommender:
    """One reranker (default λ=0.5) fitted once for the module."""
    r = RerankRecommender(retrieve_n=6, mmr_lambda=0.5)
    r.fit(catalog)
    return r


@requires_sbert
def test_recommend_similar_excludes_seed(rec: RerankRecommender) -> None:
    """recommend_similar never returns the seed itself."""
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert all(isinstance(r, Rec) for r in recs)
    assert "AEROENG C124" not in {r.course_id for r in recs}


@requires_sbert
def test_sorted_and_capped(rec: RerankRecommender) -> None:
    """Results are sorted by descending MMR score and never exceed k."""
    recs = rec.recommend_similar("AEROENG 1", k=3)
    assert len(recs) <= 3
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


@requires_sbert
def test_recommend_by_text_sorted_and_capped(rec: RerankRecommender) -> None:
    """Free-text results obey the same length + ordering contract."""
    recs = rec.recommend_by_text("aerodynamics and the physics of flight", k=4)
    assert recs
    assert len(recs) <= 4
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


@requires_sbert
def test_sparse_text_course_does_not_crash(rec: RerankRecommender) -> None:
    """A title-only seed course is reranked, not crashed on."""
    recs = rec.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


@requires_sbert
def test_empty_query_returns_empty(rec: RerankRecommender) -> None:
    """A query that normalizes to nothing yields an empty list, not garbage."""
    assert rec.recommend_by_text("   ", k=3) == []


@requires_sbert
def test_unknown_seed_raises(rec: RerankRecommender) -> None:
    """An unknown seed id raises a clear KeyError."""
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


@requires_sbert
def test_lower_lambda_increases_diversity(catalog: pd.DataFrame) -> None:
    """The phase-4 acceptance test: smaller λ ⇒ higher intra-list diversity.

    Measured in the technique-agnostic reference space (not the model's own), so
    the move is real, not self-flattering.
    """
    ref_matrix, ref_row = build_reference_space(catalog)
    query = "aerodynamics and the physics of flight"

    relevance_heavy = RerankRecommender(retrieve_n=6, mmr_lambda=1.0)
    relevance_heavy.fit(catalog)
    diversity_heavy = RerankRecommender(retrieve_n=6, mmr_lambda=0.2)
    diversity_heavy.fit(catalog)

    div_high_lambda = intra_list_diversity(
        [r.course_id for r in relevance_heavy.recommend_by_text(query, k=4)],
        ref_matrix,
        ref_row,
    )
    div_low_lambda = intra_list_diversity(
        [r.course_id for r in diversity_heavy.recommend_by_text(query, k=4)],
        ref_matrix,
        ref_row,
    )
    assert div_low_lambda >= div_high_lambda


@requires_sbert
def test_invalid_lambda_rejected() -> None:
    """An MMR λ outside [0, 1] is rejected at construction."""
    with pytest.raises(ValueError):
        RerankRecommender(mmr_lambda=1.5)


@requires_sbert
def test_invalid_retrieve_n_rejected() -> None:
    """A non-positive retrieve_n is rejected at construction."""
    with pytest.raises(ValueError):
        RerankRecommender(retrieve_n=0)
