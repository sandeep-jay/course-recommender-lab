# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

**Phase 1 (lexical baselines + harness + leaderboard) is complete and green.**
On top of the Phase 0 pipeline (package `courserec`, `config.py`/`interfaces.py`/
`data.py`, **11,073 unique courses**), the lexical rung now exists:
`recommenders/lexical.py` (`TfidfRecommender`, `BM25Recommender` with
`artifacts/<name>/` caching), `eval.py` (cross-listing + same-subject lenses,
Recall/Precision/NDCG@{5,10,20}, MAP, MRR, coverage/diversity/novelty, bootstrap
CIs on NDCG@10), and `scripts/run_eval.py` → `results/leaderboard.{md,csv}`.
`pytest` = **42 passed**; `ruff`/`black` clean. First leaderboard has **5 rows
with CIs** (acceptance met): all configs sit at NDCG@10 ≈ 0.95–0.96 with fully
overlapping CIs — no significant winner, as the methodology predicts for
near-identical cross-listed twins. `scikit-learn==1.6.1`/`scipy==1.15.1` pinned.

## Next task

**Phase 2 — Topic models** (recommender_plan.md §2.2, §5): `recommenders/topics.py`
with LSA (TruncatedSVD on TF-IDF), NMF, and LDA; recommend by similarity in
topic space; persist topic–term tables for interpretation. They drop straight
into the existing harness — add each to `build_recommenders()` in `run_eval.py`
and the leaderboard grows. Contract test per technique.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| (none open) | — | — | — |

## Blockers / waiting-on

None.

## Known gaps (deliberate, flagged per plan)

- **Free-text eval has no ground truth yet.** `recommend_by_text` works but is
  unscored — needs the hand-built judged-query set (plan §3 lens 3). Flag, don't
  silently omit.
- `docs/TRADEOFFS.md` / `docs/RESULTS.md` (cross-cutting per-phase docs) not yet
  written.

## First task for next session

Implement `src/courserec/recommenders/topics.py` (LSA first — TruncatedSVD over
the TF-IDF matrix, cosine in topic space), subclassing `Recommender`, with a
contract test; then add it to the leaderboard sweep.
