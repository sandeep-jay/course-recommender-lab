# course-rec-lab

A sandbox for learning recommender systems by implementing, documenting, and
comparing **content-based** techniques on the UC Berkeley course catalog
(~11,073 courses after cleaning). No user-interaction data exists, so
collaborative filtering is out of scope. Every technique conforms to one
interface, is scored by one shared evaluation harness, and ranked on a
leaderboard.

The full contract is [docs/roadmap/recommender_plan.md](docs/roadmap/recommender_plan.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The raw catalog CSV lives at `data/raw/courses-report_2026-06-02.csv` (gitignored
along with everything under `data/` — keep your own copy).

## Run

```bash
python scripts/prepare_data.py   # raw CSV -> data/processed/courses.parquet
pytest                           # tests
ruff check . && black .          # lint / format
```

## Status

**Phase 0 (scaffold & data) — complete.** The cleaned catalog (null-token
handling, `course_id` synthesis, level parsing, sparse-text fallback,
duplicate-id collapse) loads to a parquet of 11,073 unique courses with zero
`"-"` cells. The swappable `Recommender` interface is defined in
[src/courserec/interfaces.py](src/courserec/interfaces.py). Next: Phase 1 —
lexical baselines (TF-IDF, BM25), the eval harness, and the first leaderboard.

See [CHANGELOG.md](CHANGELOG.md) for history and [docs/adr/](docs/adr/) for
architecture decisions.
