# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

**Phases 0–3 are complete and green**, plus the judged free-text lens. Rungs:
lexical (`recommenders/lexical.py`), topic models (`recommenders/topics.py`), and
**semantic vectors** (`recommenders/embeddings.py`: `SbertRecommender` for local
MiniLM/MPNet, `ApiEmbeddingRecommender` for a hosted model, over a shared
`_EmbeddingRecommender` base with a `sha1`-keyed per-text embedding cache + FAISS
`IndexFlatIP`). The eval harness now runs **two lenses**: cross-listing
(`leaderboard.md`) and the new judged free-text set (`leaderboard_text.md`, 22
hand-labeled queries in `data/judged_queries.json`). `python scripts/run_eval.py`
regenerates both (**10 rows** each; the API row skips+flags with no key);
`pytest` = **85 passed**, `ruff`/`black` clean.

Headline: SBERT MiniLM tops **both** lenses on point estimate (cross-listing
0.971 with perfect Recall@10; free-text 0.617) — but its free-text lead over the
best TF-IDF (0.611) is **within the CI**, so semantic does not *decisively* beat
lexical on this small set. The free-text lens does cleanly separate the field
(NDCG@10 ~0.62 → ~0.07): topic models at k=50 collapse on short queries. The
semantic deps are an optional `pip install -e ".[semantic]"` extra (torch,
sentence-transformers, faiss-cpu), pinned in `pyproject.toml`.

## Next task

**Phase 4 — Retrieve + rerank + MMR** (recommender_plan.md §2.4, §5):
`recommenders/rerank.py` — a bi-encoder (reuse `SbertRecommender`) retrieves the
top ~50, a cross-encoder reranks, and MMR adds tunable-λ diversity. Acceptance:
the intra-list diversity metric moves with λ. Add to the sweep; score on both
lenses. **Alternatively / first**, strengthen the judged-query set: it is small
(22 queries) and not paraphrase-extreme, which is likely why semantic doesn't
pull away from lexical — a larger, harder set is the cheapest lever to make the
semantic advantage (or its absence) significant.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Phase 4 rerank vs. grow the judged set first | Build `rerank.py` now / expand `judged_queries.json` to ~40–60 harder queries first | Sandeep | next session |

## Blockers / waiting-on

None.

## First task for next session

Decide Phase 4 rerank vs. growing the judged set (see Open decisions). If Phase 4:
scaffold `src/courserec/recommenders/rerank.py` with a two-stage pipeline —
`SbertRecommender` (or any base) retrieves top-50, a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks, MMR re-orders with a λ knob;
contract test; add to `build_recommenders()`. The cross-encoder is another
`sentence-transformers` model, already covered by the `semantic` extra.
