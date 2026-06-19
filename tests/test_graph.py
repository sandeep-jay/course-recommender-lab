"""Contract + behavior tests for the Phase 5 graph recommender.

Beyond the shared :class:`Recommender` contract (seed excluded, ``list[Rec]`` of
length ``<= k`` sorted descending, sparse text survives), these assert the two
properties that make the graph meaningful: a directly cross-listed twin ranks
first, and — on a **held-out edge split** — a twin is recovered only when
*remaining* structure connects it (transitivity), never when its sole edge was
the one withheld. That held-out discipline is the leakage guard for the one
technique permitted to read ``Cross-Listed Course(s)`` (plan §2.6).
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.data import _build_text
from courserec.interfaces import Rec
from courserec.recommenders import graph as G
from courserec.recommenders.graph import GraphRecommender


@pytest.fixture
def catalog() -> pd.DataFrame:
    """A catalog with a cross-listed pair, a 3-way triangle, and sparse text.

    ``CS 1``/``DATA 1`` are an isolated cross-listed pair (their only edge is to
    each other). ``X 1``/``Y 1``/``Z 1`` form a fully cross-listed triangle in
    distinct subjects but a shared department. ``STAT 20`` has no description.
    """
    rows = [
        ("CS 1", "CS", "Computer Science", "Intro Programming", "Basics.", "DATA1 DS"),
        ("DATA 1", "DATA", "Computer Science", "Data Science", "Data.", "CS1 PROG"),
        ("X 1", "XX", "Shared Dept", "Topic X", "Desc x.", "Y1 TOPIC Y, Z1 TOPIC Z"),
        ("Y 1", "YY", "Shared Dept", "Topic Y", "Desc y.", "X1 TOPIC X, Z1 TOPIC Z"),
        ("Z 1", "ZZ", "Shared Dept", "Topic Z", "Desc z.", "X1 TOPIC X, Y1 TOPIC Y"),
        ("MUSIC 10", "MUSIC", "Music", "Music Theory", "Harmony.", None),
        ("STAT 20", "STAT", "Statistics", "Intro Statistics", None, None),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "course_id",
            "subject",
            "department",
            "title",
            "description",
            "cross_listed",
        ],
    )
    df["text"] = [
        _build_text(t, d) for t, d in zip(df["title"], df["description"], strict=False)
    ]
    return df.set_index("course_id")


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """Redirect the graph's artifact dir to a tmp dir (never touch the repo's)."""
    monkeypatch.setattr(G, "ARTIFACTS_DIR", tmp_path)


@pytest.fixture
def full_graph(catalog: pd.DataFrame) -> GraphRecommender:
    """The full graph (no held-out edges) fitted once for contract checks."""
    rec = GraphRecommender()
    rec.fit(catalog)
    return rec


# -- contract ------------------------------------------------------------------


def test_recommend_similar_excludes_seed(full_graph: GraphRecommender) -> None:
    """recommend_similar never returns the seed and yields Rec objects."""
    recs = full_graph.recommend_similar("X 1", k=5)
    assert all(isinstance(r, Rec) for r in recs)
    assert "X 1" not in {r.course_id for r in recs}


def test_sorted_and_capped(full_graph: GraphRecommender) -> None:
    """Results never exceed k and are sorted by descending score."""
    recs = full_graph.recommend_similar("X 1", k=2)
    assert len(recs) <= 2
    assert [r.score for r in recs] == sorted((r.score for r in recs), reverse=True)


def test_sparse_text_seed_does_not_crash(full_graph: GraphRecommender) -> None:
    """A title-only seed (no description) is handled — the graph ignores text."""
    recs = full_graph.recommend_similar("STAT 20", k=3)
    assert isinstance(recs, list)
    assert "STAT 20" not in {r.course_id for r in recs}


def test_recommend_by_text_not_implemented(full_graph: GraphRecommender) -> None:
    """The graph is item-to-item only; free text raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        full_graph.recommend_by_text("machine learning", k=3)


def test_unknown_seed_raises(full_graph: GraphRecommender) -> None:
    """An unknown seed id raises a clear KeyError."""
    with pytest.raises(KeyError):
        full_graph.recommend_similar("NOPE 999", k=3)


@pytest.mark.parametrize("restart", [0.0, 1.0, -0.1])
def test_invalid_restart_rejected(restart: float) -> None:
    """A restart probability outside (0, 1) is rejected at construction."""
    with pytest.raises(ValueError):
        GraphRecommender(restart=restart)


def test_invalid_weight_rejected() -> None:
    """A non-positive edge weight is rejected at construction."""
    with pytest.raises(ValueError):
        GraphRecommender(w_xlist=0.0)


# -- graph behavior ------------------------------------------------------------


def test_direct_twin_ranks_first(full_graph: GraphRecommender) -> None:
    """On the full graph a directly cross-listed twin is the top recommendation."""
    top = full_graph.recommend_similar("CS 1", k=3)[0]
    assert top.course_id == "DATA 1"


def test_heldout_twin_recovered_via_transitivity(catalog: pd.DataFrame) -> None:
    """A held-out edge is recovered when remaining structure still connects it.

    Hold out X–Z but keep X–Y and Y–Z (a triangle). With metadata off, the only
    paths are cross-listing edges, yet the walk reaches Z from X through Y — the
    graph's reason to exist.
    """
    held_out = frozenset({frozenset({"X 1", "Z 1"})})
    rec = GraphRecommender(use_metadata=False, held_out_edges=held_out)
    rec.fit(catalog)
    recommended = {r.course_id for r in rec.recommend_similar("X 1", k=5)}
    assert "Z 1" in recommended  # reached via the kept X–Y, Y–Z edges


def test_heldout_isolated_pair_not_recovered(catalog: pd.DataFrame) -> None:
    """A withheld edge whose endpoints share no other structure is unrecoverable.

    CS–DATA are an isolated pair; with that edge held out and metadata off, the
    seed has no edges, so the twin cannot be recovered (honest ceiling, and proof
    the held-out edge was truly removed from training).
    """
    held_out = frozenset({frozenset({"CS 1", "DATA 1"})})
    rec = GraphRecommender(use_metadata=False, held_out_edges=held_out)
    rec.fit(catalog)
    assert rec.recommend_similar("CS 1", k=5) == []


def test_metadata_glue_connects_shared_department(catalog: pd.DataFrame) -> None:
    """With metadata on, the isolated held-out pair is reconnected via shared dept.

    CS and DATA share department "Computer Science"; turning metadata back on
    surfaces the twin again through the department aux node — showing the glue's
    contribution that the structural-only baseline lacks.
    """
    held_out = frozenset({frozenset({"CS 1", "DATA 1"})})
    rec = GraphRecommender(use_metadata=True, held_out_edges=held_out)
    rec.fit(catalog)
    recommended = {r.course_id for r in rec.recommend_similar("CS 1", k=6)}
    assert "DATA 1" in recommended


def test_cache_roundtrip(catalog: pd.DataFrame) -> None:
    """A second fit loads the persisted graph and recommends identically."""
    first = GraphRecommender()
    first.fit(catalog)
    before = [(r.course_id, round(r.score, 6)) for r in first.recommend_similar("X 1")]

    second = GraphRecommender()
    second.fit(catalog)  # should hit the cache written by `first`
    after = [(r.course_id, round(r.score, 6)) for r in second.recommend_similar("X 1")]
    assert before == after
