"""The swappable recommender interface — the project's load-bearing contract.

Every technique implements :class:`Recommender` so the evaluation harness and UI
treat them identically and can be ranked side by side on one leaderboard. See
docs/roadmap/recommender_plan.md §1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Rec:
    """A single recommendation: a course and its similarity score.

    Attributes:
        course_id: The recommended course's stable id (``f"{Subject} {Course
            Number}"``, e.g. ``"AEROENG 1"``).
        score: Technique-specific similarity/relevance score. Higher is more
            relevant; values are only comparable within one technique's output,
            not across techniques.
    """

    course_id: str
    score: float


class Recommender(ABC):
    """Abstract base class every technique subclasses.

    Subclasses must set a unique ``name`` (used as the leaderboard key) and a
    ``config`` dict of hyperparameters (logged alongside results). Both
    ``recommend_*`` methods return a list of :class:`Rec` sorted by descending
    score with length ``<= k``.

    Hard rules (enforced by the contract test, violations are bugs):
        * ``recommend_similar`` MUST NOT include the seed ``course_id`` in its
          results.
        * No technique may read ``Cross-Listed Course(s)`` as an input feature —
          it is the evaluation ground truth. The graph technique is the sole
          exception and must evaluate only on a held-out edge split.
        * Sparse text (1-word or missing description) must fall back to the
          title; never crash on empty text.
    """

    name: str
    config: dict

    @abstractmethod
    def fit(self, courses: pd.DataFrame) -> None:
        """Fit the technique on the processed course catalog.

        Implementations should persist fitted artifacts (vectors, indexes,
        embedding caches) to ``artifacts/<name>/`` and load them if present
        rather than recomputing on every run.

        Args:
            courses: The processed catalog produced by
                :func:`courserec.data.load_processed` (one row per course).
        """
        raise NotImplementedError

    @abstractmethod
    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses similar to a seed course.

        Args:
            course_id: The seed course's id. Must be excluded from the results.
            k: Maximum number of recommendations to return.

        Returns:
            Up to ``k`` :class:`Rec` objects sorted by descending score, never
            including ``course_id`` itself.
        """
        raise NotImplementedError

    @abstractmethod
    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses matching a free-text query.

        Item-to-item-only techniques may raise :class:`NotImplementedError`
        here, but must never return garbage.

        Args:
            query: A natural-language query (e.g. ``"practical deep learning"``).
            k: Maximum number of recommendations to return.

        Returns:
            Up to ``k`` :class:`Rec` objects sorted by descending score.
        """
        raise NotImplementedError
