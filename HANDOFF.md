# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–4 plus the judged free-text lens are green: lexical, topic, semantic, and
now the Phase 4 retrieve→cross-encoder-rerank→MMR rung (`recommenders/rerank.py`)
all score on both lenses, with `python scripts/run_eval.py` regenerating both
leaderboards (13 rows each; the API row skips with no key) and `pytest` = **94
passed**, `ruff`/`black` clean. The judged set was grown 22 → 44 paraphrase-extreme
queries, which made the semantic win decisive — SBERT MiniLM free-text NDCG@10
0.682 (CI [0.615, 0.746]) now clears the best lexical 0.499 (CI [0.412, 0.585]) on
**non-overlapping CIs**. The MMR λ knob behaves as specified (λ 1.0→0.5→0.3 raises
intra-list diversity on both lenses), but the cross-encoder reranker does **not**
beat the bi-encoder here (domain-mismatched MS-MARCO model; twins already rank
first) — an honest, documented limitation (ADR-0005).

## Next task

**Phase 5 — graph / cross-listing-as-edges** (recommender_plan.md §2.5, §5), the
one technique allowed to use `Cross-Listed Course(s)` as input — and only on a
**held-out edge split** to avoid leakage. Build a co-listing graph, learn node
embeddings (e.g. node2vec) or use label propagation, and evaluate `recommend_*`
on the held-out edges. Alternatively, if a relevance gain from reranking is
wanted first, try a domain-tuned / different cross-encoder before moving on.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Phase 5 graph vs. revisit the cross-encoder | Build `recommenders/graph.py` on a held-out edge split / swap in a stronger or domain-tuned cross-encoder to chase a rerank relevance gain | Sandeep | next session |

## Blockers / waiting-on

None.

## First task for next session

Decide Phase 5 graph vs. revisiting the cross-encoder (see Open decisions). If
Phase 5: scaffold `src/courserec/recommenders/graph.py` — build the co-listing
graph, hold out a fraction of edges as the eval target (the sole sanctioned use of
`Cross-Listed Course(s)`), learn node embeddings, contract-test, and add to
`build_recommenders()` with its held-out-split evaluation wired into the harness.
