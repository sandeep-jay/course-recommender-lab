# Trade-offs

Technique × {quality, speed, interpretability, cost, cold-start, complexity},
with a one-line "when to prefer it". Filled in as each phase lands; for measured
numbers see [RESULTS.md](RESULTS.md) and the
[leaderboard](https://github.com/sandeep-jay/course-recommender-lab/blob/main/results/leaderboard.md).

| Technique | Quality (cross-list) | Speed | Interpretability | Cost | Cold-start | Code complexity | Prefer when |
|---|---|---|---|---|---|---|---|
| **TF-IDF + cosine** | High on near-duplicate text; blind to synonymy | Fit ~0.3 s; query ~3 ms (uni), ~32 ms (bi) | High — score is shared-term weight | None (local, no API) | Fine — needs only text | Low | A fast, explainable baseline; exact-vocabulary overlap |
| **BM25 (Okapi)** | ≈ TF-IDF here (CIs overlap); better TF saturation + length norm | Fit ~0.3 s; query ~3 ms | High — additive idf·tf terms | None (local) | Fine — needs only text | Low–medium (sparse weight matrix) | Same as TF-IDF, with length/saturation control; the stronger lexical default |
| **LSA (TruncatedSVD)** | ≈ lexical on cross-list (CIs overlap); some synonymy robustness from shared latent axes | Fit ~seconds; query ~0.3 ms (dense k=200) | Medium — signed topic–term loadings, readable but ± | None (local) | Fine — needs only text | Medium (SVD + topic-space cosine) | A compact dense reducer; a denoised cosine that still answers in sub-ms |
| **NMF** | Top point estimate (0.960) but CI overlaps all; additive parts | Fit ~seconds; query ~0.2 ms (dense k=50) | High — non-negative additive topics read as clean themes | None (local) | Fine — needs only text | Medium | Want interpretable topics + diverse lists; the most readable topic model |
| **LDA** | Lowest topic model here; weakest same-subject coherence | Fit slowest (variational, raw counts); query ~0.2 ms | High — probabilistic topic mixtures | None (local) | Fine — needs only text | Medium–high | Want a principled generative topic model / per-doc topic distributions |
| **SBERT MiniLM** (`all-MiniLM-L6-v2`) | **Top on both lenses** (xlist 0.971; text 0.682) — text lead over best lexical now **decisive** (CI [0.615,0.746] vs [0.412,0.585]); perfect xlist Recall@10 | Fit ~9 s (encode 11k on MPS); query ~0.3 ms (exact FAISS) | Low — opaque 384-d vector, no term table | None at query (local); torch dep | Fine — pretrained, needs only text | Medium–high (encoder + cache + FAISS) | Free-text / synonymy where wording differs from titles; the default semantic model |
| **SBERT MPNet** (`all-mpnet-base-v2`) | ≈ MiniLM (xlist 0.971, text 0.635); larger 768-d model didn't beat the small one here | Fit ~170 s (encode 11k); query ~0.5 ms | Low — opaque 768-d vector | None at query (local); torch dep | Fine — pretrained | Medium–high | When a bigger encoder is warranted — not demonstrably here; MiniLM is the better speed/quality trade |
| **API embeddings** (`text-embedding-3-small`) | Unmeasured — **skipped** (no key; runs local-only) | Network-bound; cost-logged | Low — opaque vector | $ per token (logged) | Fine — pretrained | Medium | A managed encoder when a key exists and local compute is constrained |
| **Rerank** (SBERT retrieve → `ms-marco-MiniLM` cross-encoder → MMR) | Did **not** beat the bi-encoder here (xlist 0.960, text 0.610 at λ=1.0 vs MiniLM 0.971/0.682) — domain-mismatched reranker, twins already rank first | Query ~70–80 ms (50 cross-encoder passes) — slowest by far | Low — cross-encoder logit + MMR trade-off score | None at query (local); torch dep | Fine — reuses base retriever | High (two stages + MMR) | You need a **diversity knob** (MMR λ moves intra-list diversity), or a domain-tuned cross-encoder is available; not for raw relevance on this catalog |
| **Graph (PPR)** (RWR over cross-listing + subject/dept aux nodes) | **Far below text** on held-out twins (NDCG@10 0.131 vs SBERT 0.913) — recovers only ~23%; isolated twin pairs are unrecoverable once their edge is withheld | Fit < 0.05 s; query ~0.4 ms (meta=off) / ~2.6 ms (meta=on) | Medium — proximity is a walk over an inspectable graph | None (local); **zero new deps** (pure `scipy.sparse`) | Poor — needs cross-listing edges; a course with none gets nothing | Medium (graph build + power-iteration RWR) | Edges encode signal **absent from text** (prereqs, sequence, co-enrollment) — *not* this catalog, where twin text is near-identical |
| **Metadata fusion** (one-hot subject+dept+level+units ⊕ TF-IDF, weight λ) | **Loses** to plain TF-IDF, monotone in λ (xlist 0.948→0.936→0.909 vs tfidf 0.955) — 99.7% of twins span subjects, so the facet block pushes them apart; text-lens identical to TF-IDF (a query has no facets) | Fit ~0.4 s; query ~ms (sparse mat-vec) | High — facet indicators + TF-IDF terms, both readable | None (local) | Fine — text + facets; sparse-text rows ride on metadata | Medium (per-block L2-norm + weighted hstack) | Browse/filter coherence ("more grad ME courses like this") — *not* the cross-listing target, which it actively hurts |
| **LLM tags** (Ollama qwen3:8b → topics/skills/prereqs, TF-IDF cosine) | **Not competitive at full (100%) enrichment:** ties lexical on xlist (0.957) and *below* tfidf on text (0.404 vs 0.461) — distilling a description to ~6–12 tags loses more detail than synonym-normalization adds; the subset-run "win" (0.960/0.585) was a vocabulary-separation artifact | Enrich ~5 s/course (one-time, cached, resumable; full catalog ~5 h); fit reads cache (fast); query ~ms + 1 Ollama call for text | High — tags are human-readable topics/skills | None (local Ollama, **no API key, zero new deps** via stdlib `urllib`) | Poor for un-enriched courses (raw-text fallback); needs an enrichment pass | Medium–high (HTTP client + tag cache + enrich pass) | Not as a ranker here — the cached tags' value is for the (future) reranker + "why this fits" UI, which use the *full* text, not lossy distillation |
| **LLM reranker** (SBERT retrieve top-20 → qwen3:8b reorders over full text) | **Measured — does not beat the base** (xlist 0.965 vs MiniLM 0.971, text 0.656 vs 0.682; both Δ negative and inside CIs, recall@10 *drops*). Reorders *full* candidate text (no distillation, unlike LLM tags) but its ceiling is the base's recall@20 — and SBERT's top-20 is already near-ceiling (twins rank first), so there's nothing to fix and only room to hurt | Query = base retrieval + 1 Ollama call (**~4 s**) on a cold cache; cached by `sha1(model+query+candidate-ids)` → warm reruns offline ~ms | Medium — yields a reordered list, no per-term reason (the "why this fits" explainer is separate) | None (local Ollama, **no API key, zero new deps**) | Fine — reuses the base retriever; needs Ollama up only on a cold cache (else falls back to base order) | High (retrieve + LLM reorder + permutation reconcile + rerank cache) | A weaker base (TF-IDF, real reorder headroom) or a larger model — **not** on top of a strong retriever at a near-ceiling task, where ~4 s/query buys nothing |

## Notes by dimension

- **Quality.** Two lenses now. On the **cross-listing** lens everything is
  statistically tied (overlapping bootstrap CIs) — SBERT has the top point
  estimate and perfect Recall@10, but the lens is near-trivial for any method
  that nails near-duplicate twin text, so it tests correctness, not quality. The
  **judged free-text** lens (now **44** hand-labeled paraphrase-extreme queries,
  see [the project plan](roadmap/recommender_plan.md)) measures `recommend_by_text`: it *discriminates* (NDCG@10 from
  ~0.68 down to ~0.06), cleanly separating lexical/semantic from the topic models,
  which collapse on short queries at k=50. On this larger, harder set SBERT MiniLM
  now beats the best TF-IDF config **decisively** (CI [0.615,0.746] vs
  [0.412,0.585], non-overlapping) — the synonymy advantage the lens was built to
  detect is real, not within-noise. Growing and hardening the query set was the
  lever that proved it.
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

- **Semantic-specific.** Embeddings trade interpretability and a heavy `torch`
  dependency for synonymy robustness and the fastest *queries* (exact FAISS over
  normalized vectors ≈ cosine, ~0.3 ms). Cold fit is the cost: encoding 11k
  courses is ~9 s (MiniLM) to ~170 s (MPNet) versus ~ms for a vectorizer — paid
  once, then a two-layer cache (per-text `sha1` store + fitted artifact) makes
  warm runs instant. The bigger MPNet did not beat MiniLM on either lens here, so
  MiniLM is the better trade. The API backend stays skipped+flagged with no key,
  honoring the local-only guarantee.

- **Rerank-specific.** The two-stage technique buys a **diversity knob**, not
  relevance: MMR λ (1.0→0.5→0.3) raises intra-list diversity monotonically on both
  lenses (xlist 0.73→0.82→0.89) while NDCG falls. The `ms-marco-MiniLM`
  cross-encoder is domain-mismatched to course text and reranks only the top 50 —
  where the bi-encoder already places the twin first — so pure rerank (λ=1.0)
  *trails* plain SBERT at ~70–80 ms/query. Value is diversity control + a hook for
  a domain-tuned cross-encoder; see [ADR-0005](adr/0005-rerank-mmr.md).

- **Graph-specific.** The graph is the one technique allowed to read
  `Cross-Listed Course(s)`, so it is scored only on a **held-out edge split**
  (its own leaderboard, [leaderboard_heldout.md](https://github.com/sandeep-jay/course-recommender-lab/blob/main/results/leaderboard_heldout.md);
  numbers there are a *harder* task than the full-truth file and not comparable
  across files). The honest result: personalized-PageRank proximity recovers only
  ~23% of withheld twins (NDCG@10 0.131) while text methods, which never needed
  the edge, score ~0.89–0.91 — twin text is near-identical, so structure adds
  nothing text didn't already have. Metadata aux nodes (`meta=on`) raise
  coverage/diversity (same-subject@10 0.00→0.82, diversity 0.01→0.87) but not
  twin recovery, since twins span subjects. A graph would pay off only on edges
  text can't see (prereqs, curricular sequence); see
  [ADR-0006](adr/0006-graph-heldout.md).

- **Clustering-specific.** Not a ranker and not in the table above — a
  **diagnostic** over the SBERT vectors (`cluster.py`), so it has no leaderboard
  row. Cost/complexity: zero new *required* deps (KMeans / Ward / HDBSCAN all from
  scikit-learn; `matplotlib`+`umap-learn` are the optional `viz` extra, and the
  map falls back from UMAP to t-SNE). Reuses the cached embeddings (no re-encode).
  Finding: forced k=100 partitions score low silhouette (~0.08–0.12) and HDBSCAN
  labels ~90% of courses noise — the space is a **smooth manifold, not tidy
  clusters** — while subject purity ~0.32 (no metadata) shows coherent-but-blended
  neighborhoods. This frames the diversity/coverage story (no hard cluster walls)
  rather than adding a ranking; see [ADR-0007](adr/0007-clustering-diagnostic.md).

- **Metadata-fusion-specific.** The fused one-hot
  subject+dept+level+units block (score `λ²·cos_text + (1−λ)²·cos_meta`, λ=1 ≡ the
  TF-IDF baseline) is a clean ablation — and the ablation says metadata *hurts*
  the cross-listing target: NDCG@10 drops monotonically as λ falls because 99.7%
  of cross-listing edges connect different subjects, so the facet block separates
  twins. On free text it is identical to TF-IDF (a query carries no facets). Its
  genuine value is browse coherence (a weak proxy the harness warns against
  optimizing) plus sparse-text robustness; see [ADR-0008](adr/0008-metadata-fusion.md).

- **LLM-specific.** A local LLM (Ollama qwen3:8b, **no API key,
  zero new deps** — stdlib `urllib`) distills each course into tags; the technique
  ranks by TF-IDF cosine over the tag profile with raw-text fallback. `fit` only
  *reads* a cache that the resumable `scripts/enrich_catalog.py` pass fills, so
  evaluation stays fast and the repo's offline guarantee holds (skips+flags when
  Ollama is down and nothing is cached). The subset run (12.5%) looked like a win
  on both lenses (xlist 0.960, text 0.585); **full-catalog enrichment overturned
  it** — at 100% coverage llm_tags ties lexical on cross-listing (0.957) and falls
  *below* plain TF-IDF on free text (0.404). The subset lift was a target/distractor
  vocabulary-separation artifact, and distilling text to a handful of tags loses
  more discriminative signal than the LLM's normalization recovers. The honest
  verdict: not competitive as a ranker — the value of the 100% tag cache is for the
  future reranker + "why this fits" UI, which use the full candidate text. See
  [ADR-0009](adr/0009-llm-enrichment-ollama.md).

- **"Why this fits" explainer (closing out the LLM enrichment track).** The two LLM ranking
  techniques both lost to SBERT (tags ADR-0009, reranker ADR-0010), and that is exactly
  the signal for where the LLM *does* pay off: not producing a ranking, but
  **justifying** one SBERT already produced. `RecommendationExplainer` takes a
  (query, recommended-candidate) pair and returns one short sentence naming the
  shared topic/skill, for the interactive Streamlit UI. It is **not** a `Recommender` and is
  **not** on the leaderboard — a free-text justification has no ground-truth
  ordering to score, so it is assessed by inspection, not a metric. Same machinery
  as the other LLM techniques (deterministic, cached by `sha1(model+query+candidate-id)`,
  zero new deps); the one difference is graceful degradation — because the line is
  *optional*, every unavailable path returns `None` (UI omits the line) instead of
  skipping a fit. Validated live (qwen3:8b) producing concrete on-topic one-liners.
  See [ADR-0011](adr/0011-llm-explainer.md).
