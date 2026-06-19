# ADR-0005: Retrieve → cross-encoder rerank → MMR diversity (Phase 4)

**Date:** 2026-06-06
**Status:** Accepted

## Context
Phase 4 adds the two-stage rung (`recommenders/rerank.py`): a fast bi-encoder
retrieves candidates, a cross-encoder reranks them, and Maximal Marginal
Relevance (MMR) re-orders for diversity (plan §2.4, §5). Several decisions were
load-bearing.

1. **Where the cross-encoder runs.** A cross-encoder scores `(query, candidate)`
   pairs jointly — accurate but `O(n)` forward passes per query, far too slow
   over all ~11k courses. It can only run over a retrieved shortlist.
2. **Diversity space.** The phase-4 acceptance test is "intra-list diversity
   moves with λ" (plan §5). Diversity is measured in a *technique-agnostic* TF-IDF
   reference space (ADR-0002); MMR must demonstrably move that external metric,
   not just an in-model one.
3. **Score monotonicity.** The interface contract requires results sorted by
   descending score. MMR selects greedily; the returned scores must still be
   non-increasing.
4. **Reuse + graceful degradation.** Retrieval should reuse the existing semantic
   rung (its cache + FAISS artifact), and the cross-encoder — another
   `sentence-transformers` model — must skip gracefully with no `semantic` extra.

## Decision
1. **Wrap a fitted base retriever; rerank at query time.** `RerankRecommender`
   holds any `_EmbeddingRecommender` (default MiniLM `SbertRecommender`), calls
   its `recommend_*` for the top `retrieve_n` (default 50), then reranks. No new
   precomputed artifact — the base persists/reloads its embeddings + FAISS index
   exactly as before; cross-encoder scoring is inherently query-time.
2. **Cross-encoder = `cross-encoder/ms-marco-MiniLM-L-6-v2`.** A small, standard
   MS-MARCO reranker from the same `semantic` extra; loaded lazily, seed-pinned
   (`torch.manual_seed(RANDOM_SEED)`) for determinism.
3. **MMR with one knob `mmr_lambda` ∈ [0, 1].** `MMR(c) = λ·rel(c) − (1−λ)·max_{s∈S}
   sim(c,s)`. `rel` is the min-max-normalized cross-encoder score (so it is
   comparable to cosine regardless of the encoder's raw scale); `sim` is cosine in
   the **bi-encoder** space (vectors already L2-normalized → a dot product). `λ=1`
   is pure cross-encoder relevance; lowering λ raises diversity.
4. **Greedy MMR returns its own marginal value as the score.** The greedy MMR
   value is provably non-increasing across selections (each pick adds to `S`, which
   can only raise the `max sim` penalty for the rest), so the emitted scores are
   sorted descending — the contract holds without a re-sort that would discard
   information.
5. **Skip gracefully.** `fit` calls `_ensure_available` first; with no extra it
   raises `EmbeddingsUnavailable`, which `run_eval.py` already catches to skip +
   flag the row.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Rerank the whole catalog (no retrieval stage) | No recall loss from retrieval | `O(11k)` cross-encoder passes per query — minutes per query | Two-stage retrieve-then-rerank is the entire point |
| MMR similarity in the reference TF-IDF space | Same space the metric uses | Would game the acceptance metric directly; not the model's own notion of similarity | Use the bi-encoder space; let the *external* metric move on its own |
| Assign rank-based descending scores | Trivially monotonic | Discards the MMR value (relevance/diversity signal) | Greedy MMR is already monotonic; keep the real values |
| Precompute a rerank artifact | Warm-run speed | Cross-encoder output depends on the (arbitrary) query — nothing to precompute | Reuse the base's artifact; rerank stays query-time |
| Bake λ into one fixed value | Fewer rows | Hides the diversity trade-off the phase is meant to show | Sweep λ ∈ {1.0, 0.5, 0.3} on the leaderboard |

## Consequences
**Positive:** The MMR knob works as specified — λ = 1.0 → 0.5 → 0.3 raises
intra-list diversity monotonically on **both** lenses (cross-listing
0.734 → 0.823 → 0.894; free-text 0.745 → 0.822 → 0.870) while NDCG@10 falls, the
phase-4 acceptance criterion. The technique drops in behind one interface and
reuses the semantic rung's cache wholesale.

**Negative (honest finding):** the cross-encoder reranker **does not beat the
bi-encoder** on this catalog/task. At λ=1.0 (pure rerank) it trails plain SBERT
MiniLM on both lenses — free-text NDCG@10 0.610 vs 0.682, cross-listing 0.960 vs
0.971. Two reasons: (a) the MS-MARCO cross-encoder is trained on web-search
query→passage relevance, a domain mismatch with course-catalog text and
especially with twin-matching; (b) bi-encoder retrieval already places the
cross-listed twin at rank 1, so reranking within the shortlist can only demote
it. Latency is ~70–80 ms/query versus sub-ms for the bi-encoder. The value
delivered here is the **diversity control**, not a relevance gain — documented,
not hidden (the repo treats limitations as first-class).

**Neutral:** A domain-tuned or fine-tuned cross-encoder might reverse the
relevance result; out of scope for this phase. MMR `sim` uses the base's private
`_embeddings`/`_row` — acceptable coupling within the package, but it ties the
reranker to embedding-style bases.

## Implementation notes
`src/courserec/recommenders/rerank.py`: `RerankRecommender` (`_ensure_available`,
lazy `_load_model`, `_cross_scores`, `_candidate_vectors`, `_mmr`, `_rerank`).
Sweep entries (λ ∈ {1.0, 0.5, 0.3}) in `scripts/run_eval.py`. Contract +
acceptance (λ↓ ⇒ diversity↑) tests in `tests/test_rerank.py`. Builds on the
semantic rung of [ADR-0004](0004-semantic-vectors.md) and the diversity space of
[ADR-0002](0002-eval-harness-design.md).
