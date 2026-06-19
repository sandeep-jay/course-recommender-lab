# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–5 plus the judged free-text lens are green; `python scripts/run_eval.py`
regenerates three leaderboards (full-truth cross-listing + judged free-text, 13
rows each; the new held-out edge lens `leaderboard_heldout.md`, 15 rows) and
`pytest` = **112 passed**, `ruff`/`black` clean. The Phase 5 graph
(`recommenders/graph.py`, the one technique allowed to read `Cross-Listed
Course(s)`) is scored only on a reproducible 30% held-out edge split and kept off
the full-truth board; on held-out twins it recovers only ~23% (NDCG@10 0.131)
versus content methods' ~0.89–0.91 (ADR-0006). The open thread is which Track-B
rung comes next — Phase 6 clustering/UMAP vs. metadata fusion.

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
