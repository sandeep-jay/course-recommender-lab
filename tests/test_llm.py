"""Contract tests for the LLM-enrichment rung (Phase 7 / Track B.8).

The LLM is never called here — a ``FakeClient`` stands in for Ollama, so the tests
need no running daemon and stay deterministic. They cover the shared
:class:`Recommender` contract plus the rung-specific behavior: ``fit`` reads the
tag cache (never generates), enrichment is resumable, raw-text fallback keeps a
cold course rankable, and the rung skips (``LLMUnavailable``) only when it would be
an empty-handed duplicate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.interfaces import Rec
from courserec.recommenders import llm as llm_mod
from courserec.recommenders.llm import (
    CourseTags,
    LLMTagRecommender,
    LLMUnavailable,
    enrich_courses,
)

_MODEL = "test-model"


class FakeClient:
    """Stand-in for :class:`OllamaClient` with no network.

    ``extract_tags`` derives topics from the text's words, so near-identical twins
    get near-identical tags (and thus high cosine). ``available`` is a flag.
    """

    def __init__(self, *, available: bool = True) -> None:
        """Configure reachability; track how many extractions were requested."""
        self.model = _MODEL
        self.host = "http://fake"
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        """Return the configured reachability flag."""
        return self._available

    def extract_tags(self, title: str, text: str) -> CourseTags:
        """Return deterministic tags derived from the text's leading words."""
        self.calls += 1
        words = [w.strip(".,").lower() for w in str(text).split()][:6]
        return CourseTags(topics=words, skills=[title.lower()], level="intro")


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Redirect both the artifact dir and the tag cache to a temp location."""
    monkeypatch.setattr(llm_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(llm_mod, "_LLMCACHE_DIR", tmp_path / "artifacts" / "llmcache")


def _enriched_rec(catalog: pd.DataFrame, **kwargs) -> LLMTagRecommender:
    """Enrich the whole catalog with a fake client, then fit a fresh recommender."""
    enrich_courses(catalog, list(catalog.index), FakeClient(), save_every=1)
    rec = LLMTagRecommender(model=_MODEL, client=FakeClient(), **kwargs)
    rec.fit(catalog)
    return rec


def test_course_tags_roundtrip_and_profile() -> None:
    """CourseTags serializes round-trip; profile excludes level, joins the rest."""
    tags = CourseTags(
        topics=["materials"],
        skills=["analyze"],
        level="grad",
        prereqs_mentioned=["calc"],
    )
    assert CourseTags.from_dict(tags.to_dict()) == tags
    profile = tags.profile_text()
    assert "materials" in profile and "analyze" in profile and "calc" in profile
    assert "grad" not in profile  # level is intentionally excluded


def test_recommend_similar_excludes_seed(mini_catalog: pd.DataFrame) -> None:
    """recommend_similar never returns the seed itself."""
    rec = _enriched_rec(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert all(isinstance(r, Rec) for r in recs)
    assert "AEROENG C124" not in {r.course_id for r in recs}


def test_recommend_similar_sorted_and_capped(mini_catalog: pd.DataFrame) -> None:
    """Results are sorted by descending score and never exceed k."""
    rec = _enriched_rec(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert len(recs) <= 3
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


def test_twin_ranks_first_from_tags(mini_catalog: pd.DataFrame) -> None:
    """Cross-listed twins share near-identical tags, so the twin tops the list."""
    rec = _enriched_rec(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert recs[0].course_id == "MATSCI C135"


def test_fit_reads_cache_never_generates(mini_catalog: pd.DataFrame) -> None:
    """Fitting reads the cache only — a fresh client sees zero calls after fit."""
    enrich_courses(mini_catalog, list(mini_catalog.index), FakeClient(), save_every=1)
    client = FakeClient()
    rec = LLMTagRecommender(model=_MODEL, client=client)
    rec.fit(mini_catalog)
    assert client.calls == 0
    assert rec._n_enriched == len(mini_catalog)


def test_enrich_is_resumable(mini_catalog: pd.DataFrame) -> None:
    """A second enrichment pass generates nothing new (all cached)."""
    ids = list(mini_catalog.index)
    first = enrich_courses(mini_catalog, ids, FakeClient(), save_every=1)
    second = enrich_courses(mini_catalog, ids, FakeClient(), save_every=1)
    assert first == len(ids)
    assert second == 0


def test_cold_cache_unavailable_skips(mini_catalog: pd.DataFrame) -> None:
    """No tags cached and Ollama down → fit raises LLMUnavailable (skip + flag)."""
    rec = LLMTagRecommender(model=_MODEL, client=FakeClient(available=False))
    with pytest.raises(LLMUnavailable):
        rec.fit(mini_catalog)


def test_cold_cache_available_falls_back_to_text(mini_catalog: pd.DataFrame) -> None:
    """No tags cached but Ollama up → fit succeeds on raw-text fallback, not a skip."""
    rec = LLMTagRecommender(model=_MODEL, client=FakeClient(available=True))
    rec.fit(mini_catalog)
    assert rec._n_enriched == 0
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert "AEROENG C124" not in {r.course_id for r in recs}


def test_sparse_text_course_does_not_crash(mini_catalog: pd.DataFrame) -> None:
    """A description-less course (title-only) is handled, not crashed on."""
    rec = _enriched_rec(mini_catalog)
    recs = rec.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


def test_recommend_by_text_enriches_query(mini_catalog: pd.DataFrame) -> None:
    """A free-text query is enriched + ranked (sorted, capped)."""
    rec = _enriched_rec(mini_catalog)
    recs = rec.recommend_by_text("composite materials heat stress", k=2)
    assert len(recs) <= 2
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


def test_recommend_by_text_offline_uses_raw_query(mini_catalog: pd.DataFrame) -> None:
    """With Ollama down, recommend_by_text falls back to the raw query, not a crash."""
    enrich_courses(mini_catalog, list(mini_catalog.index), FakeClient(), save_every=1)
    rec = LLMTagRecommender(model=_MODEL, client=FakeClient(available=False))
    rec.fit(mini_catalog)
    recs = rec.recommend_by_text("composite materials", k=2)
    assert isinstance(recs, list)


def test_unknown_seed_raises(mini_catalog: pd.DataFrame) -> None:
    """An unknown seed id raises a clear KeyError rather than returning garbage."""
    rec = _enriched_rec(mini_catalog)
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


def test_artifact_cache_roundtrips(mini_catalog: pd.DataFrame) -> None:
    """A second instance with the same config + tag cache reloads identical recs."""
    rec = _enriched_rec(mini_catalog)
    assert rec._artifact_dir.joinpath("meta.json").exists()
    reloaded = LLMTagRecommender(model=_MODEL, client=FakeClient())
    reloaded.fit(mini_catalog)
    a = rec.recommend_similar("AEROENG C124", k=3)
    b = reloaded.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in a] == [r.course_id for r in b]
