"""Technique registry and display helpers for the Phase 8 UI.

This module is the single source of truth for *which* techniques the Streamlit app
exposes, and it is intentionally free of any Streamlit import so the choices and
the formatting helpers can be unit-tested without a browser session. The UI layer
(:mod:`app.streamlit_app`) wraps these factories in Streamlit's resource cache.

The exposed set is a representative, offline, no-API-key slice of the full
leaderboard sweep (`scripts/run_eval.py`): two semantic embedders, two lexical
baselines, one topic model, and the metadata blend. Heavier or key-gated rungs
(API embeddings, the LLM tag/rerank rungs, the cross-encoder reranker, the graph
model's held-out-edge eval) are left to `scripts/run_eval.py` — the UI favours
fast, reproducible, offline retrieval. The SBERT MiniLM rung is the default
because it tops both ranking lenses (see RESULTS.md).
"""

from __future__ import annotations

from collections.abc import Callable

from courserec.interfaces import Recommender
from courserec.recommenders.embeddings import SbertRecommender
from courserec.recommenders.lexical import BM25Recommender, TfidfRecommender
from courserec.recommenders.metadata import MetadataRecommender
from courserec.recommenders.topics import LSARecommender

# Display name -> zero-arg factory. Factories are lazy so importing this module is
# cheap; the heavy fit (and any model load) happens only when the UI builds one.
TECHNIQUE_FACTORIES: dict[str, Callable[[], Recommender]] = {
    "SBERT MiniLM (top rung)": lambda: SbertRecommender(model_name="all-MiniLM-L6-v2"),
    "SBERT MPNet": lambda: SbertRecommender(model_name="all-mpnet-base-v2"),
    "TF-IDF (unigram)": lambda: TfidfRecommender(ngram_max=1, title_weight=1),
    "BM25": lambda: BM25Recommender(ngram_max=1, title_weight=1),
    "LSA (200 topics)": lambda: LSARecommender(
        n_topics=200, ngram_max=1, title_weight=1
    ),
    "Metadata + text": lambda: MetadataRecommender(text_weight=0.7),
}

# The rung shown by default — the one that wins both ranking lenses (RESULTS.md).
DEFAULT_TECHNIQUE = "SBERT MiniLM (top rung)"


def technique_names() -> list[str]:
    """Return the UI's technique display names, in registry order.

    Returns:
        The display names exposed in the technique picker. The first entry, and
        :data:`DEFAULT_TECHNIQUE`, is the recommended default.
    """
    return list(TECHNIQUE_FACTORIES)


def make_recommender(name: str) -> Recommender:
    """Instantiate (unfitted) the technique registered under ``name``.

    Args:
        name: A display name from :func:`technique_names`.

    Returns:
        A fresh, unfitted :class:`~courserec.interfaces.Recommender`. The caller
        is responsible for ``fit``.

    Raises:
        KeyError: If ``name`` is not a registered technique.
    """
    try:
        factory = TECHNIQUE_FACTORIES[name]
    except KeyError:
        raise KeyError(
            f"unknown technique: {name!r} (choices: {technique_names()})"
        ) from None
    return factory()


def course_label(course_id: str, title: object) -> str:
    """Format a course for a dropdown/result line as ``"<id> — <title>"``.

    Args:
        course_id: The course's stable id (e.g. ``"AEROENG 1"``).
        title: The course title; a missing/NA title falls back to the bare id.

    Returns:
        A single-line human-readable label.
    """
    text = "" if title is None else str(title).strip()
    if not text or text.lower() == "nan":
        return course_id
    return f"{course_id} — {text}"
