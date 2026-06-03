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
python scripts/run_eval.py       # fit + score all techniques -> results/leaderboard.{md,csv}
pytest                           # tests
ruff check . && black .          # lint / format
```

## Status

**Phase 1 (lexical baselines + harness + leaderboard) — complete.** On the
Phase 0 pipeline (cleaned catalog of 11,073 unique courses), the lexical rung now
exists: TF-IDF+cosine and Okapi BM25 ([src/courserec/recommenders/lexical.py](src/courserec/recommenders/lexical.py)),
the shared eval harness with cross-listing + same-subject lenses, full metrics,
and bootstrap CIs ([src/courserec/eval.py](src/courserec/eval.py)), and a
one-command [leaderboard](results/leaderboard.md). All lexical configs tie at
NDCG@10 ≈ 0.95–0.96 with overlapping CIs — see [docs/RESULTS.md](docs/RESULTS.md)
and [docs/TRADEOFFS.md](docs/TRADEOFFS.md). Next: Phase 2 — topic models
(LSA/NMF/LDA). The swappable `Recommender` interface is in
[src/courserec/interfaces.py](src/courserec/interfaces.py).

See [CHANGELOG.md](CHANGELOG.md) for history and [docs/adr/](docs/adr/) for
architecture decisions.
