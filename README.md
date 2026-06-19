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

**Phases 0–4 — complete.** On the Phase 0 pipeline (cleaned catalog of 11,073
unique courses), four rungs of techniques now score through one shared eval
harness ([src/courserec/eval.py](src/courserec/eval.py)) on **two lenses** —
cross-listing twins (automatic) and a hand-labeled judged free-text set (44
paraphrase-extreme queries) — with full metrics and bootstrap CIs, written to a
one-command [leaderboard](results/leaderboard.md) (13 rows per lens):

- **Phase 1 — lexical:** TF-IDF+cosine, Okapi BM25 ([lexical.py](src/courserec/recommenders/lexical.py)).
- **Phase 2 — topic models:** LSA, NMF, LDA ([topics.py](src/courserec/recommenders/topics.py)).
- **Phase 3 — semantic vectors:** local SBERT (MiniLM/MPNet) + an API backend that
  skips with no key ([embeddings.py](src/courserec/recommenders/embeddings.py)).
- **Phase 4 — retrieve→rerank→MMR:** cross-encoder rerank with an MMR diversity
  knob ([rerank.py](src/courserec/recommenders/rerank.py)).

**Headline:** SBERT MiniLM tops both lenses and now beats the best lexical config
on free text *decisively* (NDCG@10 0.682 vs 0.499, non-overlapping CIs). The
cross-encoder reranker does not beat the bi-encoder here (honest finding,
[ADR-0005](docs/adr/0005-rerank-mmr.md)); its MMR λ knob delivers tunable
diversity. See [docs/RESULTS.md](docs/RESULTS.md) and
[docs/TRADEOFFS.md](docs/TRADEOFFS.md). Next: Phase 5 — graph / cross-listing
edges. The swappable `Recommender` interface is in
[src/courserec/interfaces.py](src/courserec/interfaces.py).

See [CHANGELOG.md](CHANGELOG.md) for history and [docs/adr/](docs/adr/) for
architecture decisions.
