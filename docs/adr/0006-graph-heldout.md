# ADR-0006: Course graph (PPR) on a held-out cross-listing edge split (Phase 5)

**Date:** 2026-06-19
**Status:** Accepted

## Context
Phase 5 adds the course-graph rung (`recommenders/graph.py`, plan §2.6 / Track
B.6) — the **one technique permitted to read `Cross-Listed Course(s)` as an
input feature**. Every other method treats that column as the evaluation target;
using it elsewhere is leakage (plan §1, §3). Several decisions were load-bearing.

1. **How to stay leakage-free while reading the leak column.** The graph must be
   built from cross-listings *and* evaluated against cross-listings. Scoring it
   on the same edges it trained on would be circular.
2. **Embedding method + dependency budget.** The plan suggests node2vec. The repo
   keeps heavy deps behind extras and must run end-to-end with no API key
   (CLAUDE.md, plan §1). node2vec needs `gensim` (biased walks + skip-gram SGD) —
   a new heavy dependency and a lot of code to test.
3. **Graph density.** "Shared subject/dept" edges as same-group cliques explode:
   one 345-course subject alone is ~120k edges, and 242 subjects push into the
   millions — most of them noise.
4. **Comparability.** A graph row scored on a *different, harder* task than the
   content rows must not silently sit on the same leaderboard as if comparable.

## Decision
1. **Held-out edge split (`eval.split_crosslist_edges`).** Resolve every
   cross-listing into one undirected edge (`eval.crosslist_edges`, now the single
   source of truth for both the ground-truth builder and the graph), then split
   30% of edges out as the test target under `RANDOM_SEED`. The graph trains on
   the remaining edges; it is scored only on the held-out twins it never saw — a
   realistic link-prediction setting (a course may keep some edges while one of
   its own is withheld).
2. **Personalized PageRank (random walk with restart), not node2vec.** Rank by
   graph proximity via `r = (1−c)·P·r + c·eₛ`, solved by power iteration — pure
   `scipy.sparse`, **zero new dependencies**, deterministic, interpretable. This
   trades learned, reusable node vectors for a per-query walk; at eval scale (a
   few hundred held-out seeds × a few dozen sparse mat-vecs) the cost is
   negligible.
3. **Metadata as auxiliary nodes, not cliques.** Each course attaches to one
   node per subject and one per department (weight `w_meta`); cross-listings are
   direct course–course edges (weight `w_xlist`). Two same-subject courses are
   then two hops apart *through* the subject node — the same-group signal at
   `O(n_courses)` edges instead of `O(n²)`.
4. **A separate `leaderboard_heldout.{md,csv}`.** The graph appears only here,
   alongside the content baselines scored on the **identical** held-out edges, so
   the comparison is fair and leakage-free. A header note states this is a harder
   task than the full-truth `leaderboard.md` and the numbers are not comparable
   across files. Two graph configs (`meta=on`, `meta=off`) isolate the metadata
   glue's contribution.
5. **Item-to-item only.** The graph has no text encoder, so `recommend_by_text`
   raises `NotImplementedError`; the free-text lens skips it (and flags it),
   exactly as for any item-only technique.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| node2vec embeddings (gensim) | Canonical; reusable node vectors; could seed other tasks | New heavy dep; alias-sampling + skip-gram SGD to write & test; non-deterministic without care | PPR gives graph proximity with zero deps; revisit if reusable vectors are needed |
| Score graph on the full cross-listing truth | One leaderboard, directly comparable | Circular — trains and tests on the same edges (leakage) | Held-out split is the whole point of the leakage rule |
| Put the graph row on the main `leaderboard.md` with a note | Fewer files | Apples-to-oranges (different task) under one NDCG sort, even with a caveat | A dedicated held-out file keeps each leaderboard internally comparable |
| Same-subject/dept as group cliques | Direct same-group edges | `O(n²)` edges, millions of mostly-noise links | Auxiliary nodes give the same signal sparsely |
| Spectral / Laplacian-eigenmap embeddings | Reusable vectors, zero deps | An extra eigensolve and a less direct notion of "proximity" for link prediction | PPR is the more faithful proximity/link-prediction model here |

## Consequences
**Positive:** Phase 5 lands the sanctioned leakage exception *with* its guard —
a reproducible held-out split, a zero-dependency graph, and a fair side-by-side
leaderboard where the graph and content methods predict the same withheld edges.
The `meta=on`/`meta=off` pair makes the metadata glue's contribution legible.

**Negative / honest finding:** on the 219 held-out edges (388 seeds) the graph
recovers only **~23%** of withheld twins — NDCG@10 **0.131** (CI [0.109, 0.155]),
far below every content method (SBERT MiniLM 0.913, TF-IDF 0.895), which score
near their full-truth numbers because a held-out edge costs a text method nothing:
near-identical twin text keeps the twin at rank 1. Most cross-listings are
isolated **pairs** (mean ~1.35 twins/seed), so when a pair's only edge is the one
held out, no walk can reach it. Metadata glue (`meta=on`) does **not** lift
recovery (NDCG@10 0.130, a tie inside the CI) yet transforms list shape
(same-subject@10 0.00 → 0.82, diversity 0.01 → 0.87) — it floods the top-k with
same-subject neighbors that are rarely the cross-subject twin. The honest takeaway:
a graph adds value only when edges encode signal *absent from text* (prereqs,
sequence, co-enrollment); this catalog has none, and the held-out split exposes
that rather than letting the graph "win" by reading its own target. See
`docs/RESULTS.md` Phase 5.

**Neutral:** PPR recomputes a walk per query rather than caching node vectors;
fine at eval scale, but a UI doing thousands of live item-to-item calls might
prefer precomputed embeddings (a future lever). The graph imports
`eval.crosslist_edges`, coupling the recommender to the eval module's edge
resolution — acceptable, since that is deliberately the one shared definition of
a cross-listing edge.

## Implementation notes
`src/courserec/recommenders/graph.py`: `GraphRecommender` (`_build_adjacency`,
`_add_metadata_nodes`, `_finalize_transition`, `_rwr`, `_rank`, artifact
cache). Split + edge helpers in `src/courserec/eval.py`
(`crosslist_edges`, `_edges_to_truth`, `CrossListSplit`,
`split_crosslist_edges`). Held-out leaderboard wired in `scripts/run_eval.py`
(`build_graph_recommenders`, `_score_heldout`). Contract + held-out-behavior
tests in `tests/test_graph.py`; split tests in `tests/test_eval.py`. Builds on
the eval harness of [ADR-0002](0002-eval-harness-design.md).
