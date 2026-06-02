# Rules: src/courserec/eval.py + scripts/run_eval.py

These rules apply to the evaluation harness and leaderboard.
The methodology is `docs/roadmap/recommender_plan.md` §3.

## Leakage discipline
- When cross-listings are the target, no model may use that column as a feature.
- The graph model is evaluated only on a held-out split of cross-listing edges.

## Three lenses — never rely on one
- **Cross-listing pairs** (primary, automatic): twins should rank each other near
  the top. Caveat — near-identical text makes this trivial for lexical methods;
  it validates correctness more than quality.
- **Same-subject coherence** (weak proxy): report it as a sanity floor, never
  optimize for it (a same-subject-only model scores high while being useless).
- **Judged text-query set** (for `recommend_by_text`): the only way to evaluate
  free-text mode. If it's skipped, flag it as a gap — don't silently omit it.

## Metrics & significance
- Report Recall@k, Precision@k, MRR, MAP, NDCG@k (k ∈ {5,10,20}) plus catalog
  coverage, intra-list diversity, and novelty.
- The ground-truth set is small — report **bootstrap confidence intervals** on the
  primary metric (NDCG@10). Never crown a winner on a sub-CI gap.

## Leaderboard
- `scripts/run_eval.py` writes `results/leaderboard.{md,csv}`: one row per
  technique×config with all metrics, fit time, query latency, and API cost if any.
- Sort by NDCG@10 by default. Keep it regenerable in one command — never hand-edit.
