# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–6 plus the judged free-text lens are green; `python scripts/run_eval.py`
regenerates the three leaderboards and `python scripts/run_clustering.py`
regenerates the Phase 6 diagnostic (`results/cluster_report.{md,csv}` +
`results/plots/embedding_map.png`), with `pytest` = **125 passed** and
`ruff`/`black` clean. Phase 6 (`src/courserec/cluster.py`, a diagnostic that does
**not** subclass `Recommender`) clusters the cached SBERT vectors with
scikit-learn only (KMeans / Ward / HDBSCAN) and finds the embedding space is a
smooth manifold — forced k=100 silhouette ~0.08–0.12, HDBSCAN noises 90% of the
catalog, subject purity ~0.32 with no metadata (ADR-0007). The unbuilt Track-B
rung now is metadata fusion (plan §2.5 / B.5); Track A/B techniques remaining are
Phase 7 LLM enrichment and Phase 8 the Streamlit UI.

## Next task

**Phase 5/B.5 — metadata fusion** (recommender_plan.md §2.5, §5):
`src/courserec/recommenders/metadata.py` — one/multi-hot of subject + department +
level (and units) concatenated, **weighted**, with a text vector; sweep the
weighting and add it to the leaderboard. A ranker (subclasses `Recommender`),
unlike Phase 6. (Alternatively jump to Phase 7 LLM enrichment if a richer-feature
rung is preferred over the metadata baseline.)

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Next rung: metadata fusion vs. Phase 7 LLM enrichment | Build `recommenders/metadata.py` (one/multi-hot subject+dept+level fused with a text vector, weight-swept; joins the leaderboard) / start Phase 7 `llm.py` (LLM-extracted tags + zero-shot reranker + "why this fits", degrades gracefully with no key) | Sandeep | next session |

## Blockers / waiting-on

None.

## First task for next session

Decide metadata fusion vs. Phase 7 LLM enrichment (see Open decisions). If
metadata fusion: scaffold `src/courserec/recommenders/metadata.py` per the
`/new-recommender` contract — fuse weighted one/multi-hot subject+dept+level with
a text vector (e.g. TF-IDF or SBERT), sweep the fusion weight, persist the fitted
artifact, add a contract test, and wire it into `scripts/run_eval.py` so it lands
on the cross-listing + judged-text leaderboards.
