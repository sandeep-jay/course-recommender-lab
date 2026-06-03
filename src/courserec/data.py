"""Load and clean the UC Berkeley course catalog into a model-ready frame.

The pipeline is: read the raw CSV with pandas (descriptions contain RFC-4180
quoted newlines, so never split by line), replace the catalog's ``"-"`` null
token with real NA, synthesize a ``course_id``, parse a course level, drop dead
columns, and build the combined text signal. A handful of course_ids collide;
they are collapsed to one row each (see docs/adr/0001-duplicate-course-ids.md).
See docs/roadmap/recommender_plan.md §0.

``Cross-Listed Course(s)`` is retained in the processed frame because it is the
evaluation ground truth — but no technique may read it as an input feature.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from courserec.config import (
    NULL_TOKEN,
    PROCESSED_CATALOG_PARQUET,
    RAW_CATALOG_CSV,
)

logger = logging.getLogger(__name__)

# Raw column -> processed column. Columns absent from this map are dropped,
# which strips the dead/boilerplate columns (Terms Offered, Offering*, Repeat
# Rules, etc.) called out in the plan.
_COLUMN_RENAMES: dict[str, str] = {
    "Subject": "subject",
    "Course Number": "course_number",
    "Department(s)": "department",
    "Course Title": "title",
    "Course Description": "description",
    "Cross-Listed Course(s)": "cross_listed",
    "Credits - Units - Minimum Units": "units_min",
    "Credits - Units - Maximum Units": "units_max",
}

_LEADING_DIGITS = re.compile(r"\d+")


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows sharing a ``course_id`` to one richest representative.

    The catalog contains a handful of duplicate/near-duplicate course entries
    (see docs/adr/0001-duplicate-course-ids.md). Because ``course_id`` is the
    recommendation key and the ground-truth match key, it must be unique. We
    keep the row with the longest ``text`` (most descriptive signal, ties broken
    by first occurrence) and coalesce ``cross_listed`` across the group so no
    ground-truth cross-listing edge is dropped.

    Args:
        df: Frame indexed by ``course_id``, possibly with duplicate ids.

    Returns:
        A frame with a unique ``course_id`` index, sorted by id.
    """
    if df.index.is_unique:
        return df.sort_index()

    n_dropped = int(df.index.duplicated().sum())
    df = df.copy()
    df["cross_listed"] = df.groupby(level=0)["cross_listed"].transform(
        lambda s: s.ffill().bfill()
    )
    order = df["text"].str.len().fillna(0)
    df = df.assign(_text_len=order).sort_values("_text_len", ascending=False)
    df = df[~df.index.duplicated(keep="first")].drop(columns="_text_len")
    logger.info("Collapsed %d duplicate course_id rows", n_dropped)
    return df.sort_index()


def _course_level(course_number: str | float) -> str | None:
    """Map a raw course number to a level band.

    Berkeley course numbers may carry letter prefixes/suffixes (``N1H``,
    ``C165``); the numeric core determines the band: 1-99 lower-division,
    100-199 upper-division, 200+ graduate.

    Args:
        course_number: The raw ``Course Number`` value (may be NA).

    Returns:
        ``"lower-div"``, ``"upper-div"``, ``"grad"``, or ``None`` if no numeric
        core is present.
    """
    if not isinstance(course_number, str):
        return None
    match = _LEADING_DIGITS.search(course_number)
    if match is None:
        return None
    n = int(match.group())
    if n < 100:
        return "lower-div"
    if n < 200:
        return "upper-div"
    return "grad"


def _build_text(title: str | float, description: str | float) -> str:
    """Build the combined text signal, falling back to title for sparse rows.

    Args:
        title: Course title (100% populated in the catalog).
        description: Course description (may be NA or a single word).

    Returns:
        ``"{title}. {description}"`` when a description exists, else the title
        alone. Never raises on empty/missing text.
    """
    title_str = title if isinstance(title, str) else ""
    if isinstance(description, str) and description.strip():
        return f"{title_str}. {description}".strip()
    return title_str.strip()


def load_raw(path: Path = RAW_CATALOG_CSV) -> pd.DataFrame:
    """Read the raw catalog CSV and clean it into the processed schema.

    Args:
        path: Path to the raw catalog CSV.

    Returns:
        One row per course with synthesized ``course_id`` and ``level``, the
        null token replaced by NA, dead columns dropped, and a combined
        ``text`` field. Indexed by ``course_id``.
    """
    logger.info("Loading raw catalog from %s", path)
    raw = pd.read_csv(path, dtype=str)
    raw = raw.replace(NULL_TOKEN, pd.NA)

    df = raw[list(_COLUMN_RENAMES)].rename(columns=_COLUMN_RENAMES)

    df["course_id"] = df["subject"].str.cat(df["course_number"], sep=" ")
    df["level"] = df["course_number"].map(_course_level)
    df["units_min"] = pd.to_numeric(df["units_min"], errors="coerce")
    df["units_max"] = pd.to_numeric(df["units_max"], errors="coerce")
    df["text"] = [
        _build_text(t, d) for t, d in zip(df["title"], df["description"], strict=True)
    ]

    df = _deduplicate(df.set_index("course_id"))
    logger.info(
        "Loaded %d courses (%d cross-listed)",
        len(df),
        int(df["cross_listed"].notna().sum()),
    )
    return df


def prepare(
    raw_path: Path = RAW_CATALOG_CSV,
    out_path: Path = PROCESSED_CATALOG_PARQUET,
) -> pd.DataFrame:
    """Clean the raw catalog and persist it to parquet.

    Args:
        raw_path: Path to the raw catalog CSV.
        out_path: Destination parquet path (parent dirs are created).

    Returns:
        The processed DataFrame that was written.
    """
    df = load_raw(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("Wrote processed catalog to %s", out_path)
    return df


def load_processed(path: Path = PROCESSED_CATALOG_PARQUET) -> pd.DataFrame:
    """Load the processed catalog parquet, building it from raw if absent.

    Args:
        path: Path to the processed parquet.

    Returns:
        The processed DataFrame, indexed by ``course_id``.
    """
    if not path.exists():
        logger.info("Processed catalog missing; building from raw")
        return prepare(out_path=path)
    return pd.read_parquet(path)
