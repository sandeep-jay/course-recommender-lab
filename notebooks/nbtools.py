"""Small, tested display helpers shared by the teaching notebooks (ADR-0014).

The numbered notebooks under ``notebooks/`` reimplement each technique from
primitives for teaching and are not linted as library code. This module is the
exception: the handful of helpers they all reuse — pretty-printing a ranked
result list, measuring agreement between a from-scratch ranking and the library's,
and plotting a metric with its bootstrap CI — live here as ordinary, unit-tested,
``ruff``/``black``-clean code so the notebooks stay thin and the helpers stay
correct. Plotting depends on matplotlib (the ``notebooks`` extra); the non-plot
helpers do not import it, so they are testable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from courserec.interfaces import Rec

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard matplotlib dep
    from matplotlib.axes import Axes


def recs_to_frame(recs: Sequence[Rec], courses: pd.DataFrame) -> pd.DataFrame:
    """Render a ranked recommendation list as a readable table.

    Args:
        recs: Recommendations in rank order (as returned by a ``Recommender``).
        courses: Processed catalog indexed by ``course_id`` (for the title).

    Returns:
        A DataFrame with one row per rec — ``rank`` (1-based), ``course_id``,
        ``title`` (empty string if missing), and ``score`` — in the given order.
    """
    titles = courses["title"] if "title" in courses.columns else pd.Series(dtype=str)
    rows = [
        {
            "rank": i,
            "course_id": r.course_id,
            "title": _clean_title(titles.get(r.course_id)),
            "score": round(float(r.score), 4),
        }
        for i, r in enumerate(recs, start=1)
    ]
    return pd.DataFrame(rows, columns=["rank", "course_id", "title", "score"])


def _clean_title(title: object) -> str:
    """Coerce a possibly-missing title to a display string (``""`` when absent)."""
    if title is None:
        return ""
    text = str(title).strip()
    return "" if text.lower() == "nan" else text


def top_k_overlap(a: Sequence[str], b: Sequence[str], k: int = 10) -> float:
    """Fraction of the top-``k`` shared by two ranked ``course_id`` lists.

    The notebooks build each technique from scratch and then cross-check against
    the library version; this quantifies "do they agree" as the overlap of their
    top-``k`` sets (1.0 = identical members, order ignored).

    Args:
        a: First ranked list of ``course_id``s.
        b: Second ranked list of ``course_id``s.
        k: Cutoff for the comparison.

    Returns:
        ``|top_k(a) ∩ top_k(b)| / k`` in ``[0, 1]``. Returns 0.0 when ``k <= 0``.
    """
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / k


def plot_metric_ci(
    labels: Sequence[str],
    values: Sequence[float],
    cis: Sequence[tuple[float, float]] | None = None,
    *,
    title: str = "",
    xlabel: str = "NDCG@10",
    ax: Axes | None = None,
) -> Axes:
    """Horizontal bar chart of a metric per technique, with bootstrap CI whiskers.

    Args:
        labels: One label per bar (technique×config names).
        values: The metric value per bar (row-aligned with ``labels``).
        cis: Optional ``(low, high)`` bounds per bar for error whiskers; omit for
            no whiskers. Must be row-aligned with ``values`` when given.
        title: Optional axes title.
        xlabel: X-axis label (the metric name).
        ax: Optional existing axes to draw into; a new figure is created if None.

    Returns:
        The axes drawn into.

    Raises:
        ValueError: If ``labels``/``values`` (or ``cis``, when given) differ in length.
    """
    if len(labels) != len(values) or (cis is not None and len(cis) != len(values)):
        raise ValueError("labels, values, and cis must be the same length")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1))
    y = range(len(labels))
    xerr = None
    if cis is not None:
        xerr = [
            [v - lo for v, (lo, _) in zip(values, cis, strict=True)],
            [hi - v for v, (_, hi) in zip(values, cis, strict=True)],
        ]
    ax.barh(list(y), list(values), xerr=xerr, capsize=3, color="#4c72b0")
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(labels))
    ax.invert_yaxis()  # best (first) on top
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return ax
