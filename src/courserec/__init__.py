"""course-recommender-lab: content-based course recommenders on the UC Berkeley catalog."""

from __future__ import annotations

from courserec.config import RANDOM_SEED
from courserec.interfaces import Rec, Recommender

__all__ = ["RANDOM_SEED", "Rec", "Recommender"]

__version__ = "0.1.0"
