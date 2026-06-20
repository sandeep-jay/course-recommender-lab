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

from courserec.interfaces import Rec, Recommender
from courserec.recommenders import llm as llm_mod
from courserec.recommenders.llm import (
    CourseTags,
    LLMRerankRecommender,
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


# --------------------------------------------------------------------------- #
# Zero-shot LLM reranker (Track B.8b)                                          #
# --------------------------------------------------------------------------- #


class FakeBase(Recommender):
    """A deterministic, no-network first-stage retriever for reranker tests.

    Returns a fixed candidate list (seed excluded) in a known order, so a test can
    assert the *reranker* changed it. ``fit`` persists nothing.
    """

    def __init__(self, candidates: list[str]) -> None:
        """Configure the fixed candidate list this base always retrieves."""
        self.name = "fakebase"
        self.config = {}
        self._candidates = candidates

    def fit(self, courses: pd.DataFrame) -> None:  # noqa: D102
        self._ids = list(courses.index)

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:  # noqa: D102
        cands = [c for c in self._candidates if c != course_id][:k]
        return [Rec(c, float(len(cands) - i)) for i, c in enumerate(cands)]

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:  # noqa: D102
        cands = self._candidates[:k]
        return [Rec(c, float(len(cands) - i)) for i, c in enumerate(cands)]


class FakeRerankClient:
    """Stand-in for :class:`OllamaClient.rank_candidates` with no network.

    By default it *reverses* the candidate order, so a test can tell the rerank
    apart from the base order. ``calls`` counts live rerank requests; ``ranking``
    can be overridden to inject malformed model output.
    """

    def __init__(self, *, available: bool = True, ranking=None) -> None:
        """Configure reachability and an optional canned ranking; count calls."""
        self.model = _MODEL
        self.host = "http://fake"
        self._available = available
        self._ranking = ranking
        self.calls = 0

    def available(self) -> bool:
        """Return the configured reachability flag."""
        return self._available

    def rank_candidates(self, query: str, candidate_texts: list[str]) -> list[int]:
        """Return the canned ranking, else the reversed base order."""
        self.calls += 1
        if self._ranking is not None:
            return list(self._ranking)
        return list(range(len(candidate_texts)))[::-1]  # reverse the base order


_CANDS = ["AEROENG 1", "MUSIC 10", "MATSCI C135"]


def _rerank_rec(catalog: pd.DataFrame, client: FakeRerankClient, **kwargs):
    """Build a reranker over a FakeBase + FakeRerankClient and fit it."""
    rec = LLMRerankRecommender(
        base=FakeBase(_CANDS), model=_MODEL, client=client, **kwargs
    )
    rec.fit(catalog)
    return rec


def test_rerank_reorders_candidates(mini_catalog: pd.DataFrame) -> None:
    """The LLM order (reversed base) is what the reranker returns."""
    rec = _rerank_rec(mini_catalog, FakeRerankClient())
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in recs] == _CANDS[::-1]


def test_rerank_excludes_seed(mini_catalog: pd.DataFrame) -> None:
    """A seed that appears among base candidates is never returned."""
    rec = _rerank_rec(mini_catalog, FakeRerankClient())
    recs = rec.recommend_similar("MATSCI C135", k=3)
    assert "MATSCI C135" not in {r.course_id for r in recs}


def test_rerank_sorted_and_capped(mini_catalog: pd.DataFrame) -> None:
    """Results are sorted by descending score and never exceed k."""
    rec = _rerank_rec(mini_catalog, FakeRerankClient())
    recs = rec.recommend_similar("AEROENG C124", k=2)
    assert len(recs) <= 2
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


def test_rerank_caches_by_query_and_candidates(mini_catalog: pd.DataFrame) -> None:
    """A repeat query reuses the cached order — no second live rerank call."""
    client = FakeRerankClient()
    rec = _rerank_rec(mini_catalog, client)
    rec.recommend_similar("AEROENG C124", k=3)
    assert client.calls == 1
    rec.recommend_similar("AEROENG C124", k=3)
    assert client.calls == 1  # served from cache


def test_rerank_cold_offline_skips(mini_catalog: pd.DataFrame) -> None:
    """Ollama down and no rerank cached → fit raises LLMUnavailable (skip + flag)."""
    rec = LLMRerankRecommender(
        base=FakeBase(_CANDS), model=_MODEL, client=FakeRerankClient(available=False)
    )
    with pytest.raises(LLMUnavailable):
        rec.fit(mini_catalog)


def test_rerank_warm_offline_uses_cache(mini_catalog: pd.DataFrame) -> None:
    """A warm cache lets an offline run fit and serve the cached order."""
    online = _rerank_rec(mini_catalog, FakeRerankClient())
    online.recommend_similar("AEROENG C124", k=3)  # fills the rerank cache
    offline = _rerank_rec(mini_catalog, FakeRerankClient(available=False))
    recs = offline.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in recs] == _CANDS[::-1]


def test_rerank_uncached_offline_falls_back_to_base(mini_catalog: pd.DataFrame) -> None:
    """Warm cache for one query, but a *different* query offline → base order."""
    online = _rerank_rec(mini_catalog, FakeRerankClient())
    online.recommend_similar("AEROENG C124", k=3)  # caches only this query
    offline = _rerank_rec(mini_catalog, FakeRerankClient(available=False))
    recs = offline.recommend_similar("AEROENG 1", k=3)  # uncached query
    base_order = [c for c in _CANDS if c != "AEROENG 1"]
    assert [r.course_id for r in recs] == base_order


def test_rerank_reconciles_bad_model_output(mini_catalog: pd.DataFrame) -> None:
    """Out-of-range / partial / duplicate indices still yield a full permutation."""
    # Valid index 2, an out-of-range 9, a duplicate 2 — only 2 is usable.
    client = FakeRerankClient(ranking=[2, 9, 2])
    rec = _rerank_rec(mini_catalog, client)
    recs = rec.recommend_similar("AEROENG C124", k=3)
    ids = [r.course_id for r in recs]
    assert ids[0] == _CANDS[2]  # the one valid pick leads
    assert set(ids) == set(_CANDS)  # every candidate ranked exactly once
    assert len(ids) == len(set(ids))


def test_rerank_by_text(mini_catalog: pd.DataFrame) -> None:
    """recommend_by_text retrieves then reranks (reversed base order here)."""
    rec = _rerank_rec(mini_catalog, FakeRerankClient())
    recs = rec.recommend_by_text("composite materials", k=3)
    assert [r.course_id for r in recs] == _CANDS[::-1]


def test_rerank_empty_query_returns_empty(mini_catalog: pd.DataFrame) -> None:
    """A whitespace-only query returns no recs and triggers no rerank call."""
    client = FakeRerankClient()
    rec = _rerank_rec(mini_catalog, client)
    assert rec.recommend_by_text("   ", k=3) == []
    assert client.calls == 0


def test_rerank_unknown_seed_raises(mini_catalog: pd.DataFrame) -> None:
    """An unknown seed id raises KeyError rather than returning garbage."""
    rec = _rerank_rec(mini_catalog, FakeRerankClient())
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


def test_rerank_rejects_bad_config() -> None:
    """Non-positive retrieve_n / candidate_chars are rejected at construction."""
    with pytest.raises(ValueError):
        LLMRerankRecommender(base=FakeBase(_CANDS), retrieve_n=0)
    with pytest.raises(ValueError):
        LLMRerankRecommender(base=FakeBase(_CANDS), candidate_chars=0)
