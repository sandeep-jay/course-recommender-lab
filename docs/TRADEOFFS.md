# Trade-offs

Technique × {quality, speed, interpretability, cost, cold-start, complexity},
with a one-line "when to prefer it". Filled in as each phase lands; for measured
numbers see [RESULTS.md](RESULTS.md) and the
[leaderboard](../results/leaderboard.md).

| Technique | Quality (cross-list) | Speed | Interpretability | Cost | Cold-start | Code complexity | Prefer when |
|---|---|---|---|---|---|---|---|
| **TF-IDF + cosine** | High on near-duplicate text; blind to synonymy | Fit ~0.3 s; query ~3 ms (uni), ~32 ms (bi) | High — score is shared-term weight | None (local, no API) | Fine — needs only text | Low | A fast, explainable baseline; exact-vocabulary overlap |
| **BM25 (Okapi)** | ≈ TF-IDF here (CIs overlap); better TF saturation + length norm | Fit ~0.3 s; query ~3 ms | High — additive idf·tf terms | None (local) | Fine — needs only text | Low–medium (sparse weight matrix) | Same as TF-IDF, with length/saturation control; the stronger lexical default |

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
  mat-vec at query time.

_Topic models, semantic vectors, rerank, metadata fusion, graph, and LLM rows
land in later phases._
