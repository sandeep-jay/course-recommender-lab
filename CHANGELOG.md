# Changelog

All notable changes to course recommender implementation
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- **Phase 6 — clustering + 2-D map over the SBERT embeddings (diagnostic).**
  - `src/courserec/cluster.py` — a **diagnostic, not a `Recommender`** (no
    leaderboard row). Clusters the cached MiniLM vectors three ways with
    **scikit-learn only** (KMeans, Ward agglomerative, `sklearn.cluster.HDBSCAN`
    — no `hdbscan` package), then reports coherence: sampled cosine `silhouette`,
    `subject_purity` (size-weighted dominant-subject share — does text recover
    subjects with no metadata?), `n_noise`, and `largest_cluster_frac`. `project_2d`
    prefers UMAP and **falls back to t-SNE** when `umap-learn` is absent;
    `plot_projection` skips+flags the figure if matplotlib is missing — clustering
    runs on base+`semantic` deps, only the picture needs `viz` (ADR-0007).
  - `src/courserec/recommenders/embeddings.py` — read-only `embeddings` /
    `course_ids` properties on `_EmbeddingRecommender` so diagnostics consume the
    cached SBERT vectors without re-encoding or touching private state.
  - `scripts/run_clustering.py` — one-command driver; writes
    `results/cluster_report.{md,csv}` and `results/plots/embedding_map.png`
    (subject-colored). Skips+flags gracefully when the `semantic` extra is absent.
  - `pyproject.toml` — new optional `viz` extra (`matplotlib`, `umap-learn`);
    `src/courserec/config.py` — `PLOTS_DIR`.
  - `tests/test_cluster.py` — synthetic-blob contract tests (labels align, KMeans
    recovers blobs at purity 1.0, purity ignores noise, silhouette `nan` on a
    single cluster, t-SNE shape, plot renders, report lists algorithms). Suite:
    **125 passed** (was 112).
  - `docs/adr/0007-clustering-diagnostic.md`, `docs/RESULTS.md` Phase 6,
    `docs/TRADEOFFS.md` clustering note.
  - **Finding — the SBERT space is a smooth manifold, not tidy clusters.** Forced
    k=100 partitions score low silhouette (kmeans 0.116, agglomerative 0.082) and
    HDBSCAN labels **90% of the catalog (9,955/11,073) as noise**, keeping only 32
    dense cores; subject purity ~0.28–0.33 (no metadata) confirms coherent-but-
    blended neighborhoods. Explains why semantic similarity is graded, not
    categorical — and frames the rankers' diversity/coverage numbers.

- **Phase 5 — course graph (PPR) on a held-out cross-listing edge split.**
  - `src/courserec/recommenders/graph.py` — `GraphRecommender`: the one technique
    permitted to read `Cross-Listed Course(s)`. Ranks by personalized-PageRank
    proximity (random walk with restart, `r = (1−c)·P·r + c·eₛ`, solved by power
    iteration) over a graph of course nodes + subject/department **auxiliary
    nodes** (a star, not an `O(n²)` same-group clique). Edge weights `w_xlist`
    (cross-listings) and `w_meta` (metadata). Item-to-item only —
    `recommend_by_text` raises `NotImplementedError`. Pure `scipy.sparse`, **no
    new dependencies** (node2vec/gensim rejected — see ADR-0006). Adjacency +
    node index persist to `artifacts/<slug>/` under a fingerprint that includes
    the held-out edge set.
  - `src/courserec/eval.py` — `crosslist_edges` (the undirected/symmetric edge
    view used by the graph + split) and `_edges_to_truth`, plus `CrossListSplit`
    + `split_crosslist_edges` (reproducible 30% held-out edge split under
    `RANDOM_SEED`). `build_crosslist_truth` is left **directional and unchanged**,
    so the established Phase 1–4 cross-listing leaderboard is not perturbed (the
    edge *set* is identical either way; only the per-seed view differs).
  - `scripts/run_eval.py` — `build_graph_recommenders`, `_score_heldout`, and a
    third leaderboard `results/leaderboard_heldout.{md,csv}` where the graph and
    every content method predict the **same 219 withheld edges (388 seeds)** — a
    fair, leakage-free comparison. The graph stays **off** the full-truth
    `leaderboard.md` (scoring it there would be leakage).
  - `tests/test_graph.py` — contract tests + held-out behavior (twin recovered
    via transitivity when remaining structure connects it; isolated held-out pair
    unrecoverable; metadata reconnects via shared dept; cache round-trip).
    `tests/test_eval.py` — edge + split helper tests. Suite: **112 passed**
    (was 94).
  - `docs/adr/0006-graph-heldout.md` — design, the held-out-split leakage guard,
    node2vec-vs-PPR rationale, and the honest finding.
  - **Honest finding — on held-out twins, text crushes structure.** The graph
    recovers only ~23% of withheld twins (NDCG@10 0.131, CI [0.110, 0.156]) while
    content methods score ~0.89–0.91 (SBERT MiniLM 0.913) — a held-out edge costs
    a text method nothing, since near-identical twin text keeps the twin at rank 1.
    Most cross-listings are isolated pairs (mean ~1.35 twins/seed), so a pair's
    held-out edge is unrecoverable. Metadata glue (`meta=on`) does not lift
    recovery (0.130, tied) but raises same-subject@10 0.00 → 0.82 and diversity
    0.01 → 0.87. A graph pays off only on edges absent from text (prereqs,
    sequence) — this catalog has none. Documented, not hidden.
- **Phase 4 — retrieve → cross-encoder rerank → MMR diversity.**
  - `src/courserec/recommenders/rerank.py` — `RerankRecommender`: a MiniLM
    `SbertRecommender` retrieves the top `retrieve_n` (50), the cross-encoder
    `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores each `(query, candidate)` pair,
    then MMR re-orders with a `mmr_lambda` ∈ [0,1] knob (`λ·rel − (1−λ)·max sim`;
    `rel` = min-max-normalized cross-encoder score, `sim` = cosine in the
    bi-encoder space). Greedy MMR's value is non-increasing across picks, so the
    emitted scores stay sorted descending. No new artifact — retrieval reuses the
    base's embedding cache + FAISS index; reranking is query-time. Same `semantic`
    extra, so it skips gracefully (raises `EmbeddingsUnavailable`) with no install.
  - `scripts/run_eval.py` — three rerank rows (λ ∈ {1.0, 0.5, 0.3}) on both lenses.
  - `tests/test_rerank.py` — contract tests plus the phase-4 acceptance test
    (lower λ ⇒ higher intra-list diversity, measured in the technique-agnostic
    reference space). Suite: **94 passed** (was 85).
  - `docs/adr/0005-rerank-mmr.md` — design + the honest finding.
  - **The MMR knob works:** λ = 1.0 → 0.5 → 0.3 raises diversity monotonically on
    both lenses (cross-listing 0.734 → 0.823 → 0.894; free-text 0.745 → 0.822 →
    0.870) while NDCG@10 falls — the acceptance criterion.
  - **Honest finding — rerank does not beat the bi-encoder here.** At λ=1.0 the
    cross-encoder trails plain SBERT MiniLM on both lenses (free-text 0.610 vs
    0.682; cross-listing 0.960 vs 0.971) at ~70–80 ms/query vs sub-ms. The
    MS-MARCO cross-encoder is domain-mismatched to course text, and retrieval
    already ranks twins first, so reranking can only demote them. The value
    delivered is diversity control, not relevance — documented, not hidden.
- **Judged free-text lens expanded 22 → 44 queries (closing the significance gap).**
  - `data/judged_queries.json` — 22 new paraphrase-extreme queries across
    previously thin subjects (psychology, public health, law, art history,
    sociology, religion, development econ, etc.); 309 relevant labels over 80
    subjects (was 125 / 34). Each query deliberately avoids the words in the
    relevant courses' titles. Curated with both lexical *and* SBERT candidates so
    the labels aren't biased toward lexical matching.
  - **It made the semantic advantage significant.** On the larger, harder set the
    SBERT MiniLM free-text lead over the best lexical config is now decisive:
    NDCG@10 0.682 (CI [0.615, 0.746]) vs 0.499 (CI [0.412, 0.585]) — **non-overlapping
    CIs**, where the old 22-query set left it within the CI (0.617 vs 0.611).
- **Judged free-text lens (plan §3 lens 3) — the standing gap, now closed.**
  - `data/judged_queries.json` — 22 hand-labeled natural-language queries, 125
    relevant `course_id`s across 34 subjects, deliberately phrased to differ from
    course titles. Curated ground truth, so force-committed via a `.gitignore`
    exception (the rest of `data/` stays ignored).
  - `src/courserec/eval.py` — `JudgedQuery`, `load_judged_queries` (drops stale
    ids with a warning, skips fully-stale queries), `score_text_queries` (scores
    `recommend_by_text` with the same ranking metrics + NDCG@10 bootstrap CI;
    `same_subject@10` is NaN, undefined for free text), and
    `recommender_supports_text` (probe for item-to-item-only techniques).
    Extracted `_aggregate_ranking_metrics`, shared by both lenses.
  - `scripts/build_judged_queries.py` — `--validate` (exit-coded for CI),
    `--stats`, and `--suggest "<query>"` (lexical candidates to *seed*, not
    decide, hand labels).
  - `scripts/run_eval.py` — writes a companion `results/leaderboard_text.{md,csv}`
    sorted by NDCG@10; flags any skipped/unavailable technique in the header.
  - **It discriminates where cross-listing couldn't:** free-text NDCG@10 spreads
    from ~0.62 (lexical/SBERT) to ~0.07 (NMF/LDA at k=50), which collapse on short
    queries. The free-text mode that topic/semantic methods are meant to win is
    finally measurable.
- **Phase 3 — semantic vectors (SBERT local + API embeddings).**
  - `src/courserec/recommenders/embeddings.py` — dense-vector rankers over a
    shared `_EmbeddingRecommender` base: `SbertRecommender` (Sentence-Transformers
    `all-MiniLM-L6-v2` 384-d and `all-mpnet-base-v2` 768-d, MPS on Apple Silicon
    else CPU) and `ApiEmbeddingRecommender` (hosted model, token+cost logging).
    Two-layer cache: a per-text store keyed by `sha1(model_name + normalized_text)`
    at `artifacts/embcache/<model>/` (a text encoded at most once per model, ever),
    plus the fitted artifact (`embeddings.npy`, `course_ids.json`, `index.faiss`,
    `meta.json`). Search is exact FAISS `IndexFlatIP` over L2-normalized vectors
    (inner product = cosine; `index_type="hnsw"` exposes approximate ANN), so eval
    stays deterministic.
  - **Graceful degradation:** torch/sentence-transformers/faiss-cpu are an optional
    pinned `semantic` extra, imported lazily; the API backend raises
    `EmbeddingsUnavailable` (no SDK or no key) which `run_eval.py` catches to skip +
    flag the row — the suite still runs local-only with no key.
  - `scripts/run_eval.py` — sweep extended with both SBERT models + the API model;
    leaderboard grows to **10 rows** (API skipped, flagged).
  - Tests: `tests/test_embeddings.py` (10 cases — contract per the SBERT backend,
    semantic free-text match, artifact roundtrip, invalid index_type, API
    graceful-skip), plus 4 lens tests in `tests/test_eval.py`. Suite: **85 tests**.
  - Results: SBERT MiniLM tops **both** lenses on point estimate (xlist 0.971 with
    perfect Recall@10; text 0.617) — but its free-text lead over the best TF-IDF
    (0.611) is **within the CI**, so semantic does not decisively beat lexical on
    this 22-query set. The larger MPNet did not beat MiniLM. Honest finding,
    pointing at a bigger/paraphrase-heavier query set and rerank (Phase 4).
  - ADRs: [0003](docs/adr/0003-judged-query-lens.md) (judged-query lens),
    [0004](docs/adr/0004-semantic-vectors.md) (semantic vectors, caching, ANN).
- **Phase 2 — topic models (LSA / NMF / LDA).**
  - `src/courserec/recommenders/topics.py` — three latent-topic rankers over a
    shared `_TopicRecommender` base: `LSARecommender` (TruncatedSVD on TF-IDF),
    `NMFRecommender` (non-negative factorization of TF-IDF), `LDARecommender`
    (variational LDA over raw counts). Each reduces the catalog to a dense topic
    space and scores by cosine (the doc–topic matrix is L2-normalized once at fit,
    so similarity is a plain dot product). Persists `vectorizer.pkl`, `model.pkl`,
    `doc_topics.npy`, `topic_terms.json`, `meta.json` to `artifacts/<name>/` with a
    class-aware corpus+config fingerprint (loads if present). Topic→term
    interpretation tables exposed via `topic_terms()` and previewed in the fit log.
  - Heavy documentation/comments/logging per the new plan mandate: module sketch
    (idea, math, complexity, wins/loses), Google-style docstrings, and structured
    `INFO` logging of corpus shape, cache hit/miss, model diagnostics (LSA explained
    variance, NMF reconstruction error, LDA perplexity), and a topic preview.
  - `scripts/run_eval.py` — sweep extended with LSA(k=200), NMF(k=50), LDA(k=50);
    leaderboard grows to **8 rows**.
  - Tests: `tests/test_topics.py` (29 cases — contract per technique, OOV-query →
    empty, cross-class cache isolation, config validation, topic-table populated).
    Suite: **71 tests**.
  - Fixes found while wiring: guarded denormal-small row norms in
    `_l2_normalize_rows` (would overflow to ±inf and NaN-poison the matmul), and
    silenced spurious BLAS `matmul` FP-warnings on Apple Silicon via `np.errstate`
    (matrices verified finite first).
  - Results: NMF tops at NDCG@10 = 0.960 but its CI overlaps every other
    technique — no significant winner, as the methodology predicts for
    near-identical cross-listed twins. LDA is interpretable but lowest, with
    notably weaker same-subject coherence (0.079 vs ~0.19 for lexical). The
    judged-query lens (still missing) remains the real test for topic models.
- **Documentation/comments/logging mandate** added to
  `docs/roadmap/recommender_plan.md` §5 cross-cutting — module docstrings,
  Google-style method docstrings, why-not-what inline comments, and structured
  `logging` of the eval lifecycle are now non-negotiable for every technique.
- **Phase 1 — lexical baselines + harness + leaderboard.**
  - `src/courserec/recommenders/lexical.py` — `TfidfRecommender` (cosine over
    L2-normalized TF-IDF) and `BM25Recommender` (Okapi BM25 folded into a
    precomputed sparse doc-term weight matrix), sharing a base that handles
    title-weighting, sparse-text fallback, seed exclusion, and `artifacts/<name>/`
    persistence keyed by a corpus+config fingerprint (load if present).
  - `src/courserec/eval.py` — cross-listing ground truth (resolves the
    space-stripped `cross_listed` references; 1,072 in-catalog seeds) and
    same-subject sanity floor; Recall/Precision/NDCG@{5,10,20}, MAP, MRR,
    catalog coverage, intra-list diversity (in a technique-agnostic reference
    space), novelty; percentile bootstrap CIs on NDCG@10.
  - `scripts/run_eval.py` — fits the lexical sweep (stopwords/n-grams/title
    weight), scores all, writes `results/leaderboard.{md,csv}` sorted by NDCG@10
    in one command.
  - Tests: `tests/test_lexical.py` (contract: seed excluded, sorted, ≤k,
    sparse-text safe, twin ranks first, artifact cache round-trips) and
    `tests/test_eval.py` (truth resolution, hand-checked metrics, bootstrap,
    end-to-end). Shared synthetic catalog in `tests/conftest.py`. Suite: 42 tests.
  - First leaderboard: all lexical configs land at NDCG@10 ≈ 0.95–0.96 with
    fully overlapping 95% CIs — no significant winner, exactly as the methodology
    warns (cross-listed twins share near-identical text). Real signal is latency
    (BM25 ~3 ms/query vs bigram TF-IDF ~33 ms).
  - Pinned `scikit-learn==1.6.1` and `scipy==1.15.1` in `pyproject.toml`.
  - `docs/RESULTS.md` (Phase 1 interpretation + honest limitations) and
    `docs/TRADEOFFS.md` (technique × {quality, speed, interpretability, cost,
    cold-start, complexity} matrix); README status/run updated for `run_eval.py`.
  - `docs/adr/0002-eval-harness-design.md` — decision to resolve cross-listing
    ground truth by space-stripped token lookup and to measure intra-list
    diversity in a fixed, technique-agnostic TF-IDF reference space.
- **Phase 0 — scaffold & data.** `pyproject.toml` (package `course-rec-lab`,
  pinned pandas/numpy/pyarrow + dev tools, ruff/black/pytest config); `src/courserec/`
  with `config.py` (paths, `RANDOM_SEED=42`, `NULL_TOKEN`), `interfaces.py`
  (`Rec`, `Recommender` ABC with the seed-exclusion / no-leakage contract), and
  `data.py` (load → null-token→NA → `course_id` → level parsing → sparse-text
  fallback → duplicate-id collapse → parquet).
- `scripts/prepare_data.py` — one-command raw CSV → `data/processed/courses.parquet`.
- `tests/test_data.py` — 18 tests: row count, no `"-"` remains, id synthesis +
  uniqueness, level bands, sparse-text fallback, cross-listed sparsity.
- `docs/adr/0001-duplicate-course-ids.md` + ADR index — decision to collapse the
  16 colliding `course_id`s (34 rows) to one representative each, coalescing
  `cross_listed`. Catalog: 11,091 raw rows → **11,073 unique courses**.
- `README.md` (setup, run, Phase-0 status).
- Project-specific Claude Code rules: `.claude/rules/recommenders.md` (interface
  contract, seed exclusion, no-leakage, artifacts) and `.claude/rules/eval.md`
  (leakage discipline, three lenses, bootstrap CIs, regenerable leaderboard).
- `/new-recommender` command — scaffolds a technique + contract test + scoring,
  replacing the lakehouse `/new-transform`.
- `.gitignore` (artifacts, `*.npz`, `data/`, `.env*`, `.claude/`,
  Python/tooling/OS noise). Initialized git repo on branch `main`.

### Changed
- Rewrote `CLAUDE.md` for the course recommender (5 load-bearing rules, run
  commands, data notes) — was an empty scribe-iq-lakehouse template.
- Trimmed `.claude/settings.json` to pytest/ruff/black/git/fs allows; dropped
  AWS-S3 and detect-secrets entries.
- Reset `.claude/settings.local.json` to an empty allow list (kept additionalDirectories).
- Slimmed `.claude/hooks/scan-secrets.sh` to AWS + generic patterns; added
  OpenAI/Anthropic key patterns; dropped Azure/OneLake/MLflow/DagsHub/Fabric.
- Retargeted `/session-start` (Session-1 fallback → recommender_plan.md) and
  `/session-end` (doc-sync → leaderboard/TRADEOFFS/RESULTS).
- Fixed `HANDOFF.md` title.
- Untracked `.claude/` and `data/` from git and broadened `.gitignore` to ignore
  both wholesale (was only `settings.local.json` and `data/processed/`).
- Moved the catalog CSV to its canonical path `data/raw/courses-report_2026-06-02.csv`
  (resolving the plan-vs-disk discrepancy); chose `course-rec-lab` as the package name.

### Removed
- Lakehouse leftovers copied from scribe-iq-lakehouse: `.claude/rules/{fabric-transforms,
  notebooks,transforms}.md`, `.claude/skills/{delta-patterns,healthcare-data}.md`,
  `.claude/commands/new-transform.md`.

