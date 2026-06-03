# ADR-0004: Semantic-vector recommenders (SBERT local + API), caching and ANN

**Date:** 2026-06-03
**Status:** Accepted

## Context
Phase 3 adds the semantic rung (`recommenders/embeddings.py`): dense vectors from
a pretrained encoder, the first technique that can match synonymy/paraphrase the
lexical and low-rank topic methods miss (plan §2.3, §5). Several decisions were
load-bearing and easy to get wrong.

1. **Dependency weight.** SBERT pulls `torch` (hundreds of MB). The repo must run
   end-to-end Phases 0–2 *without* it, and Phases 0–6 with **no API key** (plan §1).
2. **Caching.** Re-encoding 11k courses every run is slow; re-encoding via an API
   would also cost money (plan §6.6). The rule mandates an embedding cache keyed
   by `sha1(model_name + normalized_text)`.
3. **ANN index.** The plan asks for FAISS/hnswlib "overkill at 11k but part of
   the learning" — but an approximate index makes the eval non-deterministic.
4. **API graceful degradation.** An API backend must never hard-fail the suite
   when no key is present.

## Decision
1. **Two backends behind one base class.** `_EmbeddingRecommender` owns docs,
   caching, normalization, the index, persistence, and both `recommend_*`
   methods; subclasses supply only the encoder. `SbertRecommender` runs
   Sentence-Transformers locally (MPS on Apple Silicon, else CPU; fp32 per the
   Apple-Silicon rule), with `all-MiniLM-L6-v2` (384-d) and `all-mpnet-base-v2`
   (768-d) in the sweep. `ApiEmbeddingRecommender` calls a hosted model and logs
   token count + dollar cost.
2. **Optional `semantic` extra.** `torch`/`sentence-transformers`/`faiss-cpu` are
   a pinned `[project.optional-dependencies] semantic` extra, not base deps;
   `embeddings.py` imports them lazily inside methods so the module imports fine
   without them. Absence raises `EmbeddingsUnavailable`.
3. **Two-layer cache.** Layer 1 is a content-addressed per-text store keyed by
   `sha1(model_name + normalized_text)` at `artifacts/embcache/<model>/`, shared
   across configs/runs so a text is encoded at most once per model, ever. Layer 2
   is the fitted-recommender artifact (`artifacts/<name>/`: normalized matrix,
   `course_ids`, the FAISS index, a corpus fingerprint) so a warm run reloads
   without re-encoding or re-indexing.
4. **Exact FAISS `IndexFlatIP` by default; HNSW available.** Vectors are
   L2-normalized so inner product *is* cosine. The default exact flat index keeps
   the eval deterministic and is instant at 11k; `index_type="hnsw"` exposes the
   approximate ANN index for the learning exercise without making the leaderboard
   non-reproducible.
5. **Graceful skip.** With no key (or no SDK), the API backend raises
   `EmbeddingsUnavailable`, which `run_eval.py` catches to skip + flag the row —
   the suite still runs local-only. Keys come only from an env var, never a
   literal (no-hardcoded-secrets rule).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Make torch/SBERT base dependencies | Simpler install story | Forces a heavy download on anyone running Phases 0–2; breaks the light base install | Optional extra + lazy import keeps the base light |
| HNSW (approximate) as the default index | "Real" ANN; sub-linear queries | Non-deterministic recall → flaky eval; needless at 11k | Flat is exact and instant here; HNSW kept as an option |
| Single artifact only (no per-text cache) | Less code | Re-encodes shared text across configs; can't reuse work between MiniLM configs | The sha1 per-text cache is a rule and avoids real recompute/cost |
| Cosine via a brute-force numpy matmul | No FAISS dep | Misses the plan's ANN learning goal; reinvents search | FAISS is the point of the exercise and gives both flat + HNSW |
| Lowercase text before hashing/encoding | Smaller cache key space | The encoders are cased; meaning can ride on case | Normalize whitespace only, preserve case |

## Consequences
**Positive:** A drop-in semantic rung scoring on both lenses; deterministic eval;
no recompute on warm runs; runs local-only with no key; the API path is ready and
cost-instrumented for when a key exists. SBERT now leads both lenses on point
estimate and is the only method with perfect cross-listing Recall@10 (1.000).
**Negative:** Heavy dependency (torch) and slower cold fit (mpnet ~170 s to encode
11k vs ~ms for a vectorizer; MiniLM ~9 s). Embeddings are opaque versus a
topic–term table. On the judged free-text lens SBERT's lead over the best lexical
config is **within the CI** — semantic does not *decisively* beat lexical on this
small query set (see [RESULTS.md](../RESULTS.md)); honest, and a pointer to a
larger/paraphrase-heavier query set and to rerank (Phase 4).
**Neutral:** The API backend is unexercised in CI (no key); its graceful-skip
path is unit-tested, its live path is not.

## Implementation notes
`src/courserec/recommenders/embeddings.py`: `_EmbeddingCache` (layer-1 sha1
cache), `_EmbeddingRecommender` (base), `SbertRecommender`,
`ApiEmbeddingRecommender`, `EmbeddingsUnavailable`. Pinned extra in
`pyproject.toml`. Sweep + graceful-skip wiring in `scripts/run_eval.py`. Contract
+ skip tests in `tests/test_embeddings.py`. Builds on the free-text lens of
[ADR-0003](0003-judged-query-lens.md).
