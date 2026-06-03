# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

**Phase 0 (scaffold & data) is complete and green.** The package `courserec`
exists under `src/` with `config.py` (paths, `RANDOM_SEED`, `NULL_TOKEN`),
`interfaces.py` (`Rec`, `Recommender` ABC), and `data.py` (full clean pipeline).
`scripts/prepare_data.py` writes `data/processed/courses.parquet` — **11,073
unique courses, 0 `"-"` cells, 242 subjects, 1,080 cross-listed**. `pytest` =
18 passed; `ruff`/`black` clean. Open decisions resolved: canonical CSV path is
`data/raw/courses-report_2026-06-02.csv`, package name is `course-rec-lab`;
`.claude/` and `data/` are gitignored. All committed on `main` (not pushed);
the one open thread is starting Phase 1.

## Next task

**Phase 1 — Lexical baselines + harness + leaderboard** (recommender_plan.md §5):
1. `src/courserec/recommenders/lexical.py` — TF-IDF+cosine and BM25, both
   subclassing `Recommender`; small config sweep (stopwords, n-grams, title weight).
2. `src/courserec/eval.py` — cross-listing + same-subject lenses, metrics
   (Recall/Precision/MRR/MAP/NDCG@{5,10,20}, coverage, diversity, novelty),
   bootstrap CIs on NDCG@10.
3. `scripts/run_eval.py` — fit all, score all, write `results/leaderboard.{md,csv}`.
4. Contract tests per technique (seed excluded, sorted, ≤k, sparse-text safe).
   *Accept:* one command produces a leaderboard with ≥2 rows and CIs.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| (none open) | — | — | — |

## Blockers / waiting-on

None.

## First task for next session

Implement `src/courserec/recommenders/lexical.py` (TF-IDF + cosine, then BM25)
against the `Recommender` interface, with a contract test.
