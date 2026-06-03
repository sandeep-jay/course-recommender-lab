"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest
from courserec.data import _build_text


@pytest.fixture
def mini_catalog() -> pd.DataFrame:
    """A tiny synthetic catalog exercising twins, distinct topics, and sparse text.

    ``AEROENG C124`` and ``MATSCI C135`` are cross-listed twins (near-identical
    text); ``STAT 20`` has no description (sparse-text fallback to title).
    """
    rows = [
        (
            "AEROENG C124",
            "AEROENG",
            "C124",
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation.",
            "MATSCIC135 MATS EXTRM ENVRMTS",
        ),
        (
            "MATSCI C135",
            "MATSCI",
            "C135",
            "Materials in Extreme Environments",
            "Composite materials under heat stress and radiation loads.",
            "AEROENGC124 MAT EXTRM ENVRMTS",
        ),
        (
            "AEROENG 1",
            "AEROENG",
            "1",
            "Introduction to Flight",
            "Aerodynamics, lift, drag, and the principles of flight.",
            None,
        ),
        (
            "MUSIC 10",
            "MUSIC",
            "10",
            "Introduction to Music Theory",
            "Harmony, counterpoint, melody, and rhythm.",
            None,
        ),
        (
            "STAT 20",
            "STAT",
            "20",
            "Introduction to Statistics",
            None,  # sparse: no description -> falls back to title
        ),
    ]
    df = pd.DataFrame(
        [r + (None,) if len(r) == 5 else r for r in rows],
        columns=[
            "course_id",
            "subject",
            "course_number",
            "title",
            "description",
            "cross_listed",
        ],
    )
    df["text"] = [
        _build_text(t, d) for t, d in zip(df["title"], df["description"], strict=False)
    ]
    return df.set_index("course_id")
