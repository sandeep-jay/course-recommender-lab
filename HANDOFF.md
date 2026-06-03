# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

**Phases 0–2 are complete and green:** the data pipeline, the swappable
`Recommender` interface, the eval harness, and two technique rungs — lexical
(`recommenders/lexical.py`: TF-IDF, BM25) and topic models
(`recommenders/topics.py`: LSA, NMF, LDA, all with `artifacts/<name>/` caching
and topic–term tables). `python scripts/run_eval.py` regenerates an **8-row**
leaderboard; `pytest` = **71 passed**, `ruff`/`black` clean. Headline: NMF leads
NDCG@10 at 0.960 but every technique's CI overlaps — no significant winner,
because the cross-listing lens rewards near-identical twin text and so can't
separate the methods; the one open thread is that `recommend_by_text` is still
**unscored** for lack of the judged-query set (plan §3 lens 3).

## Next task

**Phase 3 — Semantic vectors** (recommender_plan.md §2.3, §5): `recommenders/embeddings.py`
with SBERT (`all-MiniLM-L6-v2` + one larger model) locally, then an API embedding
model behind the same interface; embedding cache keyed by
`sha1(model_name + normalized_text)`; FAISS/hnswlib ANN. Must run local-only with
no API key (API path degrades gracefully). Contract test per technique, add to the
sweep. **Alternatively**, close the standing gap first: build the judged-query set
(`scripts/build_judged_queries.py` + a `recommend_by_text` lens in `eval.py`) so
free-text mode — the thing topic/semantic methods are supposed to win — is finally
measurable.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| (none open) | — | — | — |

## Blockers / waiting-on

None.

## First task for next session

Decide Phase 3 vs. the judged-query gap (see Next task), then start. If Phase 3:
scaffold `src/courserec/recommenders/embeddings.py` with the SBERT
`all-MiniLM-L6-v2` model behind `Recommender`, embedding cache keyed by
`sha1(model_name + normalized_text)`, cosine over normalized vectors; contract
test; add to `build_recommenders()`. Confirm `sentence-transformers` is a pinned
dep first (it is not yet in `pyproject.toml`).
