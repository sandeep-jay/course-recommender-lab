"""Contract tests for the metadata-fusion recommender (Phase 5 / Track B.5).

Beyond the shared :class:`Recommender` contract (seed excluded, sorted, capped,
sparse-text safe), these pin the fusion-specific behavior: the ``text_weight``
knob bounds correctly, ``λ = 1`` recovers pure TF-IDF, metadata pulls
same-facet courses up, and a description-less course still ranks via its metadata
block. They run against a small synthetic catalog carrying real facets.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.data import _build_text
from courserec.interfaces import Rec
from courserec.recommenders import metadata as metadata_mod
from courserec.recommenders.lexical import TfidfRecommender
from courserec.recommenders.metadata import MetadataRecommender


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Redirect artifact persistence to a temp dir so tests never touch the repo."""
    monkeypatch.setattr(metadata_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")


@pytest.fixture
def meta_catalog() -> pd.DataFrame:
    """A synthetic catalog with the structured facets metadata fusion consumes.

    ``AEROENG C124`` / ``MATSCI C135`` are cross-listed twins (near-identical
    text, *different* subject/department). ``AEROENG 200`` shares AEROENG's
    subject + department + grad level but is topically unrelated — the lever that
    separates metadata from text. ``STAT 20`` is sparse (title-only description).
    """
    rows = [
        (
            "AEROENG C124",
            "AEROENG",
            "Aerospace Engineering",
            "grad",
            4.0,
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation.",
        ),
        (
            "MATSCI C135",
            "MATSCI",
            "Materials Science",
            "grad",
            4.0,
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation loads.",
        ),
        (
            "AEROENG 200",
            "AEROENG",
            "Aerospace Engineering",
            "grad",
            3.0,
            "Graduate Seminar in Flight Dynamics",
            "Orbital mechanics, control, and trajectory optimization.",
        ),
        (
            "MUSIC 10",
            "MUSIC",
            "Music",
            "lower-div",
            2.0,
            "Introduction to Music Theory",
            "Harmony, counterpoint, melody, and rhythm.",
        ),
        (
            "STAT 20",
            "STAT",
            "Statistics",
            "lower-div",
            3.0,
            "Introduction to Statistics",
            None,
        ),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "course_id",
            "subject",
            "department",
            "level",
            "units_min",
            "title",
            "description",
        ],
    )
    df["text"] = [
        _build_text(t, d) for t, d in zip(df["title"], df["description"], strict=True)
    ]
    return df.set_index("course_id")


def test_recommend_similar_excludes_seed(meta_catalog: pd.DataFrame) -> None:
    """recommend_similar never returns the seed itself."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(meta_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=4)
    assert all(isinstance(r, Rec) for r in recs)
    assert "AEROENG C124" not in {r.course_id for r in recs}


def test_recommend_similar_sorted_and_capped(meta_catalog: pd.DataFrame) -> None:
    """Results are sorted by descending score and never exceed k."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(meta_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert len(recs) <= 3
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_text_weight_must_be_in_unit_interval() -> None:
    """A fusion weight outside [0, 1] is rejected at construction."""
    with pytest.raises(ValueError):
        MetadataRecommender(text_weight=1.5)


def test_lambda_one_matches_pure_tfidf(meta_catalog: pd.DataFrame) -> None:
    """text_weight=1.0 ignores metadata, ranking identically to plain TF-IDF."""
    fused = MetadataRecommender(text_weight=1.0, title_weight=1, ngram_max=1)
    fused.fit(meta_catalog)
    tfidf = TfidfRecommender(title_weight=1, ngram_max=1)
    tfidf.fit(meta_catalog)
    a = [r.course_id for r in fused.recommend_similar("AEROENG C124", k=4)]
    b = [r.course_id for r in tfidf.recommend_similar("AEROENG C124", k=4)]
    assert a == b


def test_metadata_pulls_same_facet_course_up(meta_catalog: pd.DataFrame) -> None:
    """Metadata weight ranks a same-facet course above an unrelated one."""
    rec = MetadataRecommender(text_weight=0.1)
    rec.fit(meta_catalog)
    ranked = [r.course_id for r in rec.recommend_similar("AEROENG C124", k=4)]
    # AEROENG 200 shares subject+department+level (3 facets) with the seed;
    # MUSIC 10 shares no facet and little text, so it scores ~0 and ranks below
    # (or drops out entirely). Metadata-dominant fusion must surface AEROENG 200.
    assert "AEROENG 200" in ranked
    if "MUSIC 10" in ranked:
        assert ranked.index("AEROENG 200") < ranked.index("MUSIC 10")


def test_sparse_text_course_does_not_crash(meta_catalog: pd.DataFrame) -> None:
    """A description-less course (title-only) is handled via its metadata block."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(meta_catalog)
    recs = rec.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


def test_recommend_by_text_ignores_metadata(meta_catalog: pd.DataFrame) -> None:
    """Free-text query returns sorted, capped recs (text block only)."""
    rec = MetadataRecommender(text_weight=0.7)
    rec.fit(meta_catalog)
    recs = rec.recommend_by_text("composite materials heat", k=2)
    assert len(recs) <= 2
    assert all(isinstance(r, Rec) for r in recs)
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


def test_unknown_seed_raises(meta_catalog: pd.DataFrame) -> None:
    """An unknown seed id raises a clear KeyError rather than returning garbage."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(meta_catalog)
    with pytest.raises(KeyError):
        rec.recommend_similar("NOPE 999", k=3)


def test_missing_facet_columns_do_not_crash(mini_catalog: pd.DataFrame) -> None:
    """A catalog lacking department/level/units still fits (subject-only metadata)."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(mini_catalog)
    recs = rec.recommend_similar("AEROENG C124", k=3)
    assert isinstance(recs, list)
    assert all(isinstance(r, Rec) for r in recs)


def test_artifact_cache_roundtrips(meta_catalog: pd.DataFrame) -> None:
    """A second instance with the same config loads from the persisted artifact."""
    rec = MetadataRecommender(text_weight=0.5)
    rec.fit(meta_catalog)
    assert rec._artifact_dir.joinpath("meta.json").exists()
    reloaded = MetadataRecommender(text_weight=0.5)
    reloaded.fit(meta_catalog)
    a = rec.recommend_similar("AEROENG C124", k=3)
    b = reloaded.recommend_similar("AEROENG C124", k=3)
    assert [r.course_id for r in a] == [r.course_id for r in b]
