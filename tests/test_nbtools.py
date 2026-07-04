"""Contract tests for the notebook display helpers (``notebooks/nbtools.py``).

These keep the shared helpers honest without executing any notebook: the
notebooks reuse them, so a regression here would silently corrupt every
teaching breakdown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from courserec.interfaces import Rec
from notebooks import nbtools


@pytest.fixture
def courses() -> pd.DataFrame:
    """A tiny catalog-shaped frame indexed by course_id, with a missing title."""
    return pd.DataFrame(
        {"title": ["Intro Stats", "Deep Learning", None]},
        index=["STAT 1", "COMPSCI 182", "MISC 1"],
    )


def test_recs_to_frame_shape_order_and_missing_title(courses: pd.DataFrame) -> None:
    """Renders a ranked table preserving order; a missing title becomes ``""``."""
    recs = [Rec("COMPSCI 182", 0.92), Rec("MISC 1", 0.5)]
    frame = nbtools.recs_to_frame(recs, courses)
    assert list(frame.columns) == ["rank", "course_id", "title", "score"]
    assert frame["rank"].tolist() == [1, 2]  # 1-based, preserves input order
    assert frame.loc[0, "course_id"] == "COMPSCI 182"
    assert frame.loc[0, "title"] == "Deep Learning"
    assert frame.loc[1, "title"] == ""  # missing title -> empty string, no crash


def test_recs_to_frame_empty(courses: pd.DataFrame) -> None:
    """An empty rec list yields an empty frame with the columns still present."""
    frame = nbtools.recs_to_frame([], courses)
    assert len(frame) == 0
    assert list(frame.columns) == ["rank", "course_id", "title", "score"]


def test_top_k_overlap_identical_disjoint_and_partial() -> None:
    """Overlap is 1.0 for identical sets, 0.0 for disjoint, fractional between."""
    a = ["a", "b", "c", "d"]
    assert nbtools.top_k_overlap(a, a, k=4) == 1.0
    assert nbtools.top_k_overlap(a, ["w", "x", "y", "z"], k=4) == 0.0
    assert nbtools.top_k_overlap(a, ["a", "b", "z", "y"], k=4) == 0.5
    # order within the top-k is ignored; only membership counts
    assert nbtools.top_k_overlap(["a", "b"], ["b", "a"], k=2) == 1.0


def test_top_k_overlap_nonpositive_k() -> None:
    """A non-positive cutoff is a degenerate 0.0, never a divide-by-zero."""
    assert nbtools.top_k_overlap(["a"], ["a"], k=0) == 0.0


def test_plot_metric_ci_draws_one_bar_per_label() -> None:
    """One horizontal bar and one y-tick label is drawn per technique."""
    matplotlib = pytest.importorskip("matplotlib")  # optional viz/notebooks dep

    matplotlib.use("Agg")  # headless, no GUI backend
    labels = ["tfidf", "bm25"]
    values = [0.8, 0.7]
    cis = [(0.75, 0.85), (0.65, 0.75)]
    ax = nbtools.plot_metric_ci(labels, values, cis, title="t", xlabel="NDCG@10")
    assert len(ax.patches) == len(labels)  # one bar per technique
    assert [t.get_text() for t in ax.get_yticklabels()] == labels


def test_plot_metric_ci_length_mismatch_raises() -> None:
    """Mismatched labels/values lengths fail loudly rather than mis-plotting."""
    with pytest.raises(ValueError, match="same length"):
        nbtools.plot_metric_ci(["a", "b"], [0.1])
