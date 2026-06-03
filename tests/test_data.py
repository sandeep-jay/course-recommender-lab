"""Contract tests for the data loader (Phase 0 acceptance).

These assert the load-bearing data invariants: row count, the ``"-"`` null
token is gone, ``course_id`` synthesis, level parsing, and sparse-text
fallback.
"""

from __future__ import annotations

import pandas as pd
import pytest
from courserec.config import RAW_CATALOG_CSV
from courserec.data import _build_text, _course_level, load_raw


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    """The cleaned catalog, loaded once for the module."""
    if not RAW_CATALOG_CSV.exists():
        pytest.skip(f"raw catalog not present at {RAW_CATALOG_CSV}")
    return load_raw()


def test_row_count(catalog: pd.DataFrame) -> None:
    """The catalog has ~11,091 rows; 18 are dropped as duplicate course_ids.

    The raw CSV has 11,091 rows but 16 course_ids appear on 34 rows (see
    docs/adr/0001-duplicate-course-ids.md); after collapsing each to one
    representative, 11,073 unique courses remain.
    """
    assert len(catalog) == 11073


def test_no_null_token_remains(catalog: pd.DataFrame) -> None:
    """No cell still holds the literal ``"-"`` null token."""
    assert not (catalog == "-").to_numpy().any()


def test_course_id_synthesis(catalog: pd.DataFrame) -> None:
    """course_id is ``f"{subject} {course_number}"`` and is the index."""
    assert catalog.index.name == "course_id"
    row = catalog.iloc[0]
    assert catalog.index[0] == f"{row['subject']} {row['course_number']}"


def test_course_ids_unique(catalog: pd.DataFrame) -> None:
    """Synthesized course ids are unique (safe to use as an index/key)."""
    assert catalog.index.is_unique


def test_cross_listed_is_sparse(catalog: pd.DataFrame) -> None:
    """Cross-listed ground truth is retained and ~10% populated."""
    populated = catalog["cross_listed"].notna().mean()
    assert 0.05 < populated < 0.15


@pytest.mark.parametrize(
    ("course_number", "expected"),
    [
        ("1", "lower-div"),
        ("N1H", "lower-div"),
        ("99", "lower-div"),
        ("100", "upper-div"),
        ("C165", "upper-div"),
        ("199", "upper-div"),
        ("200", "grad"),
        ("602", "grad"),
        ("", None),
        (None, None),
    ],
)
def test_course_level(course_number: str | None, expected: str | None) -> None:
    """Level bands are parsed from the numeric core of the course number."""
    assert _course_level(course_number) == expected


def test_levels_cover_catalog(catalog: pd.DataFrame) -> None:
    """Every level value is one of the three known bands (or NA)."""
    valid = {"lower-div", "upper-div", "grad"}
    assert set(catalog["level"].dropna().unique()) <= valid


def test_text_fallback_to_title() -> None:
    """Missing/empty description falls back to the title, never crashes."""
    assert _build_text("Intro to Flight", pd.NA) == "Intro to Flight"
    assert _build_text("Intro to Flight", "") == "Intro to Flight"
    assert _build_text("Intro to Flight", "Aerodynamics.").startswith(
        "Intro to Flight. Aerodynamics"
    )


def test_text_never_empty_when_titled(catalog: pd.DataFrame) -> None:
    """Text signal is non-empty for every titled course."""
    titled = catalog[catalog["title"].notna()]
    assert (titled["text"].str.len() > 0).all()
