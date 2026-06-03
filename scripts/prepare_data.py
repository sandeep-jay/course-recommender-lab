#!/usr/bin/env python3
"""Build the processed course catalog parquet from the raw CSV.

Usage:
    python scripts/prepare_data.py

Reads the raw catalog (``data/raw/``), cleans it via
:func:`courserec.data.prepare`, and writes ``data/processed/courses.parquet``.
Regenerable in one command; the processed output is gitignored.
"""

from __future__ import annotations

import logging

from courserec.config import PROCESSED_CATALOG_PARQUET, RAW_CATALOG_CSV
from courserec.data import prepare


def main() -> None:
    """Run the data-preparation pipeline and report a short summary."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    df = prepare(raw_path=RAW_CATALOG_CSV, out_path=PROCESSED_CATALOG_PARQUET)
    logging.getLogger(__name__).info(
        "Done: %d courses, %d subjects -> %s",
        len(df),
        df["subject"].nunique(),
        PROCESSED_CATALOG_PARQUET,
    )


if __name__ == "__main__":
    main()
