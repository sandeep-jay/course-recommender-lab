# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

**Phase 2 (topic models: LSA / NMF / LDA) is complete and green.** On top of the
Phase 1 lexical rung + harness, `recommenders/topics.py` now adds three
latent-topic rankers over a shared `_TopicRecommender` base — `LSARecommender`
(TruncatedSVD on TF-IDF), `NMFRecommender` (non-negative factorization),
`LDARecommender` (variational LDA over raw counts) — all scoring by cosine in a
once-normalized dense topic space, with `artifacts/<name>/` caching (class-aware
fingerprint), persisted topic–term interpretation tables, and structured logging
(corpus shape, cache hit/miss, model diagnostics, topic preview) per the new
documentation/logging mandate in plan §5. `scripts/run_eval.py` fits them too, so
the leaderboard is now **8 rows**. `pytest` = **71 passed**, `ruff`/`black` clean.
NMF tops NDCG@10 at 0.960 but its CI overlaps every technique — still no
significant winner (near-identical twins). Two fixes landed while wiring:
denormal row-norm guard in `_l2_normalize_rows`, and `np.errstate` to silence
spurious Apple-Silicon BLAS `matmul` warnings (matrices verified finite).

Open thread unchanged and now more pressing: `recommend_by_text` works for every
technique but is **still unscored** — the judged-query set (plan §3 lens 3) is the
only way to show topic models' real payoff (synonym/paraphrase robustness), since
the cross-listing lens can't distinguish them from lexical.

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
