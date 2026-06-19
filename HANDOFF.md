# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–5 plus the judged free-text lens are green: lexical, topic, semantic,
rerank, and now the Phase 5 graph rung (`recommenders/graph.py`) all run, with
`python scripts/run_eval.py` regenerating **three** leaderboards — full-truth
cross-listing + judged free-text (13 rows each) and the new held-out edge lens
(`leaderboard_heldout.md`, 15 rows: 13 content + 2 graph) — and `pytest` = **112
passed**, `ruff`/`black` clean. The graph (the one technique allowed to read
`Cross-Listed Course(s)`) is scored only on a reproducible 30% **held-out edge
split** and stays off the full-truth leaderboard. Honest finding: on held-out
twins the graph recovers only ~23% (NDCG@10 0.131) while content methods score
~0.89–0.91 — twin text is near-identical, so structure adds nothing text didn't
already have; metadata aux nodes raise coverage/diversity but not twin recovery
(ADR-0006).

## Next task

**Phase 6 — clustering + UMAP** (recommender_plan.md §2.7, §5): KMeans /
agglomerative / HDBSCAN over the SBERT embeddings, a UMAP/t-SNE 2-D plot colored
by subject saved to `results/plots/`. Diagnostic, not a ranker — feeds the
coverage/diversity analysis. (Metadata fusion, plan §2.5 Track B.5, is the other
unbuilt Track-B rung if a ranker is preferred over a diagnostic next.)

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Phase 6 clustering/UMAP vs. metadata fusion next | Build `cluster.py` (diagnostic + UMAP plot) / build `recommenders/metadata.py` (one/multi-hot subject+dept+level fused with a text vector, weight-swept) | Sandeep | next session |

## Blockers / waiting-on

None.

## First task for next session

Decide Phase 6 clustering/UMAP vs. metadata fusion (see Open decisions). If
Phase 6: scaffold `src/courserec/cluster.py` — cluster the SBERT embeddings
(KMeans/agglomerative/HDBSCAN), fit UMAP, save a subject-colored 2-D plot to
`results/plots/`, and surface cluster-coherence/coverage numbers; it is a
diagnostic, so it does not subclass `Recommender` or join the leaderboard.
