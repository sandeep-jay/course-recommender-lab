# Trade-offs

Technique × {quality, speed, interpretability, cost, cold-start, complexity},
with a one-line "when to prefer it". Filled in as each phase lands; for measured
numbers see [RESULTS.md](RESULTS.md) and the
[leaderboard](../results/leaderboard.md).

| Technique | Quality (cross-list) | Speed | Interpretability | Cost | Cold-start | Code complexity | Prefer when |
|---|---|---|---|---|---|---|---|
| **TF-IDF + cosine** | High on near-duplicate text; blind to synonymy | Fit ~0.3 s; query ~3 ms (uni), ~32 ms (bi) | High — score is shared-term weight | None (local, no API) | Fine — needs only text | Low | A fast, explainable baseline; exact-vocabulary overlap |
| **BM25 (Okapi)** | ≈ TF-IDF here (CIs overlap); better TF saturation + length norm | Fit ~0.3 s; query ~3 ms | High — additive idf·tf terms | None (local) | Fine — needs only text | Low–medium (sparse weight matrix) | Same as TF-IDF, with length/saturation control; the stronger lexical default |
| **LSA (TruncatedSVD)** | ≈ lexical on cross-list (CIs overlap); some synonymy robustness from shared latent axes | Fit ~seconds; query ~0.3 ms (dense k=200) | Medium — signed topic–term loadings, readable but ± | None (local) | Fine — needs only text | Medium (SVD + topic-space cosine) | A compact dense reducer; a denoised cosine that still answers in sub-ms |
| **NMF** | Top point estimate (0.960) but CI overlaps all; additive parts | Fit ~seconds; query ~0.2 ms (dense k=50) | High — non-negative additive topics read as clean themes | None (local) | Fine — needs only text | Medium | Want interpretable topics + diverse lists; the most readable topic model |
| **LDA** | Lowest topic model here; weakest same-subject coherence | Fit slowest (variational, raw counts); query ~0.2 ms | High — probabilistic topic mixtures | None (local) | Fine — needs only text | Medium–high | Want a principled generative topic model / per-doc topic distributions |

## Notes by dimension

- **Quality.** On the cross-listing lens all lexical configs are statistically
  tied (overlapping bootstrap CIs). The lens is near-trivial for lexical methods
  because twins share text; it tests correctness, not quality. The synonymy
  blind spot ("ML" vs "machine learning") is the gap semantic vectors must beat.
- **Speed.** Bigram TF-IDF is ~10× slower per query than unigram/BM25 for no
  NDCG gain — the vocabulary explosion is pure cost here.
- **Interpretability.** Both are bag-of-words: a recommendation is explainable as
  the terms two courses share. BM25 weights are additive idf·tf contributions.
- **Cost / cold-start.** Fully local, no API key, no training data beyond the
  catalog text; handle any course with a title (sparse-text rows fall back to
  title). Fitted vectors persist to `artifacts/<name>/` and reload on next run.
- **Complexity.** TF-IDF is one scikit-learn `fit_transform`. BM25 adds a custom
  sparse doc-term weight matrix (`bm25_weight_matrix`) but stays a single
  mat-vec at query time. Topic models add a factorization step (SVD / NMF / LDA)
  and store a dense, L2-normalized doc–topic matrix so the query is again one
  mat-vec — but dense over `k` topics, which is *faster* than the lexical sparse
  mat-vec here (~0.2 ms vs ~3 ms).
- **Diversity.** Topic models return more varied top-k lists (intra-list
  diversity ~0.71–0.83) than the lexical methods (~0.74), and NMF/LDA in
  particular pull in fewer same-subject neighbours (same-subject@10 ~0.08–0.11
  vs ~0.19) — they generalise past exact vocabulary into shared themes. Whether
  that is *better* recommendation can't be judged on the cross-listing lens.

_Semantic vectors, rerank, metadata fusion, graph, and LLM rows land in later
phases._
