"""Tests for the Phase 8 UI glossary (`app/glossary.py`).

The glossary is the explanatory layer the UI renders: a blurb per exposed technique,
a paragraph per family, a definition per leaderboard metric, and the three eval
lenses. These tests pin the contracts the UI relies on — every picker technique has a
blurb, every real leaderboard name resolves to a known family, every metric column is
defined — so a new technique or metric can't silently ship without its explanation.
Streamlit is never imported, so they run in the base environment.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import glossary, registry
from courserec.config import RESULTS_DIR

LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"

# The metric columns the Leaderboard view shows tooltips for (non-metric columns
# name/family/config are handled separately).
_NON_METRIC = {"name", "family"}


def test_every_exposed_technique_has_a_blurb() -> None:
    """Each registry technique the picker offers must have a description."""
    for name in registry.technique_names():
        assert name in glossary.TECHNIQUE_INFO, f"missing blurb for {name!r}"
        assert glossary.TECHNIQUE_INFO[name].strip()


@pytest.mark.parametrize(
    "name",
    [
        "sbert(all_minilm_l6_v2,idx=flat)",
        "tfidf(sw=on,ng=1-1,tw=1)",
        "bm25(sw=on,ng=1-1,tw=1,k1=1.5,b=0.75)",
        "lsa(k=200,sw=on,ng=1-1,tw=1)",
        "nmf(k=50,sw=on,ng=1-1,tw=1)",
        "lda(k=50,sw=on,ng=1-1,tw=1)",
        "metadata(text=0.7,facets=subject+department+level+units)",
        "rerank(cross_encoder_ms_marco_minilm_l_6_v2,base=sbert,n=50,mmr=0.5)",
        "llm_tags(qwen3_8b)",
        "llm_rerank(qwen3_8b,base=sbert,n=20)",
        "graph(use_metadata=True)",
    ],
)
def test_family_of_resolves_known_prefixes(name: str) -> None:
    """Every real technique-name prefix maps to a labelled family, not 'other'."""
    family = glossary.family_of(name)
    assert family != "other", f"{name!r} fell through to 'other'"
    assert family in glossary.FAMILIES
    assert glossary.family_label(name) == glossary.FAMILIES[family]


def test_family_of_unknown_is_other() -> None:
    """An unrecognized prefix degrades to the 'other' family, never raises."""
    assert glossary.family_of("mystery(x=1)") == "other"
    assert glossary.family_label("mystery(x=1)") == glossary.FAMILIES["other"]


def test_every_family_key_has_a_description() -> None:
    """FAMILIES and FAMILY_DESCRIPTIONS stay in lockstep."""
    assert set(glossary.FAMILIES) == set(glossary.FAMILY_DESCRIPTIONS)
    assert all(v.strip() for v in glossary.FAMILY_DESCRIPTIONS.values())


def test_metric_help_known_and_unknown() -> None:
    """metric_help returns a definition for a metric and None otherwise."""
    assert glossary.metric_help("ndcg@10")
    assert glossary.metric_help("not_a_column") is None


@pytest.mark.skipif(
    not LEADERBOARD_CSV.exists(), reason="leaderboard.csv not generated"
)
def test_glossary_covers_the_actual_leaderboard() -> None:
    """Every metric column and technique name in the real leaderboard is explained."""
    board = pd.read_csv(LEADERBOARD_CSV)
    for col in board.columns:
        if col not in _NON_METRIC:
            assert glossary.metric_help(col), f"no glossary entry for column {col!r}"
    for name in board["name"]:
        assert glossary.family_of(name) != "other", f"unmapped technique {name!r}"
