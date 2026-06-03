"""Project-wide constants and filesystem paths.

Centralizing paths here keeps the rest of the codebase free of hardcoded
locations (per the project's no-hardcoded-paths rule) and gives every module a
single import for the random seed.
"""

from __future__ import annotations

from pathlib import Path

# Global seed for any stochastic step, so runs are reproducible.
RANDOM_SEED: int = 42

# The catalog's null sentinel. Anything equal to this string is missing data,
# never a real value. See docs/roadmap/recommender_plan.md §0.
NULL_TOKEN: str = "-"

# Repo layout. config.py lives at src/courserec/config.py, so the project root
# is three parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Canonical input and output for the data pipeline.
RAW_CATALOG_CSV: Path = RAW_DIR / "courses-report_2026-06-02.csv"
PROCESSED_CATALOG_PARQUET: Path = PROCESSED_DIR / "courses.parquet"
