# Changelog

All notable changes to course recommender implementation
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- **Documentation site — MkDocs Material published to GitHub Pages (ADR-0015).**
  A navigable site matching the portfolio's two published repos: `mkdocs.yml`
  (Material, `strict: true`, mermaid, `edit_uri`), `requirements-docs.txt`, a `[docs]`
  extra, and `.github/workflows/docs.yml` (build `--strict` + Pages-artifact deploy —
  one-time Settings → Pages → Source = "GitHub Actions"). Five new reviewer/learner
  pages (`index`, `reviewer-guide`, `ARCHITECTURE`, `case-study`, `about`) plus a
  `notebooks/` index and a `changelog` that includes the repo-root `CHANGELOG.md`
  verbatim; existing RESULTS/TRADEOFFS/RUNBOOK/ADRs/roadmap wire into the nav.
- **Teaching notebooks rendered as executed notebooks on the site (ADR-0015).**
  `scripts/render_notebooks.py` (+ a `Makefile` `docs-notebooks` target) executes the
  `notebooks/*.py` sources locally against the real catalog (and live Ollama for `08`)
  and writes pre-executed `.ipynb` to `docs/notebooks/`, which `mkdocs-jupyter` renders
  with `execute: false` — CI never runs a notebook. A scoped exception to ADR-0014:
  the `.py` percent script stays the source of truth; the committed `.ipynb` are a
  generated publish artifact (`.gitignore` narrowed from `*.ipynb` to `notebooks/*.ipynb`).

### Changed
- **Renamed the project to `course-recommender-lab`** across the distribution name,
  README/HANDOFF/RUNBOOK/roadmap titles, the package docstring, `CLAUDE.md`, and the
  Docker image tag (was `course-rec-lab` / `course-rec-bert` / `course-rec-ui`). The
  Python import package stays `courserec` (hyphens are illegal in module names).

- **Teaching notebooks — ten step-by-step breakdowns under `notebooks/` (ADR-0014).**
  The teaching companion to the library: one notebook per technique family, each
  *reimplementing the method from primitives on the real catalog* (reusing the
  library only for plumbing — data load + the eval harness) and **running its own
  eval live**, bracketed by a data/eval foundation and a cross-technique synthesis.
  - `00_data_and_eval` (the `"-"` null, leakage, the 3 lenses, the metrics —
    definitions; first live numbers in `01`), `01_lexical` (TF-IDF + BM25 + the
    preprocessing sweep), `02_topics` (LSA via SVD; NMF/LDA), `03_embeddings` (SBERT
    + cosine/FAISS retrieval by hand; the synonym test lexical fails), `04_rerank`
    (cross-encoder + MMR from scratch), `05_metadata` (one-hot ⊕ text fusion —
    honest negative), `06_graph` (adjacency + power-iterated PPR + the held-out
    eval), `07_clustering` (KMeans/HDBSCAN + the 2-D map), `08_llm` (tags / rerank /
    explainer via local Ollama, degrading gracefully), `09_leaderboard` (the
    canonical board + bootstrap-CI synthesis).
  - **Lean teaching style:** explain each stage → run it on the real catalog →
    important transformations in their own cells → cross-check the from-scratch
    ranking against the library → live eval (slow rungs sample seeds). No toy
    corpora or exercises.
  - **Tooling:** new `notebooks` extra (jupytext, nbmake, ipykernel, matplotlib).
    The **`.py` percent scripts are the versioned source** (clean diffs); the
    `.ipynb` are generated and gitignored; **nbmake** executes them (opt-in
    `--nbmake`, kept out of the default suite). `notebooks/nbtools.py` holds the
    shared display helpers (`recs_to_frame`, `top_k_overlap`, `plot_metric_ci`) —
    ordinary linted, unit-tested code (`tests/test_nbtools.py`, +6 tests → 211).
    Numbered notebooks are excluded from `ruff`/`black`; `notebooks/README.md`
    indexes them and the RUNBOOK gains a **Notebooks (4b)** section.
- **Deploy — warm, offline CPU Docker image for the UI (ADR-0013).** A `Dockerfile`
  packages the Phase 8 Streamlit UI so a fresh host starts **warm** (no first-load
  encode) and **offline** (no runtime network):
  - Bakes in the processed catalog, `results/`, the six UI rungs' artifacts + the Map
    projection, and the **default MiniLM weights** (pre-pulled at build). So
    course-to-course, Compare, Map, the Leaderboard, and **free-text queries** all
    work with no network. MPNet's weights and the Ollama why-line stay on-demand
    (graceful degradation, unchanged).
  - Forces the **CPU torch wheel** (installs `torch==2.12.0` from the PyTorch CPU
    index before `pip install -e ".[ui,semantic]"`), so the multi-GB CUDA default is
    never pulled. CPU-only image.
  - `$PORT`-aware `CMD` + headless `.streamlit/config.toml`, runs as a non-root
    `appuser`, `/_stcore/health` HEALTHCHECK — so Cloud Run / HF Spaces (Docker) work
    unchanged. `.dockerignore` ships only the runtime subset (~150 MB of artifacts;
    raw CSV, regenerable caches, `tests/`/`scripts/`/`docs/`/`notebooks/`, and unused
    rungs excluded — the Dockerfile's targeted `COPY`s never pull them, this keeps the
    build context lean too).
  - Honest caveat: the build `COPY`s the gitignored `data/processed/` + `artifacts/`,
    so it must run from a **warm repo** (pipeline + eval already run) — not a fresh
    clone. Documented in the RUNBOOK's new **Deploy (4a)** section + troubleshooting
    rows; `README.md` gains a one-command build/run.
- **`docs/RUNBOOK.md` — operational runbook.** End-to-end how-to-run for the whole
  repo: install tiers (`dev`/`semantic`/`viz`/`ui` extras + recommended combos), the
  data pipeline, every script with its exact flags and outputs (`run_eval`,
  `run_clustering`, `enrich_catalog`, `explain_recs`, `prepare_data`,
  `build_judged_queries`), the Streamlit UI, the Ollama setup, a **models reference
  table** (every SBERT/cross-encoder/API/Ollama model — source, how to get it, what
  it needs), the artifact/cache map, and a troubleshooting table for the common
  failure modes (run-from-root, missing extras, Ollama down, cold caches). Linked
  from `README.md` and `CLAUDE.md`.
- **Phase 8 — minimal Streamlit UI (plan §4).** Surfaces every prior phase from one
  `streamlit run`: the techniques, the Phase 7c explainer, and the full leaderboard.
  - `app/registry.py` — the import-safe, **Streamlit-free** core: `TECHNIQUE_FACTORIES`
    (six representative offline rungs — SBERT MiniLM (default), SBERT MPNet, TF-IDF,
    BM25, LSA, Metadata+text), `make_recommender` (`KeyError` on unknown),
    `course_label` (`"<id> — <title>"`, falls back to the bare id on a missing title).
    Curated subset by design (no API/LLM/cross-encoder/graph rungs) — fast, offline,
    no key; the full sweep stays in `scripts/run_eval.py` and the Leaderboard table.
  - `app/projection.py` + a fourth **Map view** — a live, interactive 2-D projection
    of the SBERT catalog (Altair: hover for the course, scroll-zoom, pan). Pick a seed
    and its top-k SBERT recommendations light up (🔴 seed / 🔺 recs / grey rest), so
    you can *see* where recommendations land relative to the seed. UMAP/t-SNE toggle
    (UMAP needs `.[viz]`, else t-SNE). Projecting ~11k vectors is the slow step, so
    `app/projection.py` (Streamlit-free, tested) computes the layout once and caches
    it to `artifacts/map/` keyed by `(method, model, seed)` — recompute on a shape
    mismatch, never serve a stale layout. ~12 s cold (t-SNE), instant warm.
  - `app/glossary.py` — an **explanatory layer** (same Streamlit-free, tested pattern
    as the registry) so users running Compare or reading the leaderboard understand
    what they see: a one-line blurb per exposed technique, a paragraph per *family*
    (so even leaderboard rows the UI never fits are explained), a definition per
    leaderboard metric, and the three eval lenses + the leakage guardrail.
    `family_of`/`family_label` map a raw technique name (`sbert(…)`, `bm25(…)`) to a
    family; `metric_help` powers the per-column header tooltips. Descriptions stay
    honest to the findings (e.g. metadata fusion *hurting* cross-listing is stated).
    Wired in: a blurb under each technique picker (Explore + both Compare columns); a
    `family` column, hover-for-definition column tooltips, and "How to read this
    leaderboard" / "Technique families" expanders on the Leaderboard.
  - Leaderboard view also renders the **graph technique's held-out-edge board**
    (`results/leaderboard_heldout.csv`) as a clearly-labeled second table — "not
    comparable to the table above" — so the graph results (ADR-0006) are visible in
    the UI, not only on disk. The `graph(…)` rows sort to the bottom (~0.23 recall@10
    vs content's ~1.0 on the same split): holding out a twin's edge isolates it in
    the graph, while content methods read its near-identical text directly. A test
    asserts every held-out name maps to a known `family` (so the column can't break).
  - `app/streamlit_app.py` — four views: **Explore** (course or free-text → top-k
    with scores + an opt-in "why this fits" column), **Compare** (one query, two
    techniques side by side), **Leaderboard** (the main board + the graph's held-out
    board + the static map), and **Map** (the live interactive projection). Catalog,
    each fitted technique, the explainer, and the projection are
    `st.cache_resource`-cached, so interactions never re-fit; the
    why-line reuses `RecommendationExplainer` (ADR-0011) and degrades to `—` when
    Ollama is down. New optional extra `ui` (`streamlit==1.41.1`);
    `pip install -e ".[ui,semantic]"`.
  - `tests/test_app_registry.py` (14) + `tests/test_app_glossary.py` (17: every
    exposed technique has a blurb, every real name in *both* leaderboards — graph
    rows included — resolves to a known family, every metric column is defined,
    `family_of` unknown → `"other"`) + `tests/test_app_projection.py` (4: cold key
    computes + caches, warm key reuses without recomputing, shape-mismatch recomputes,
    path encodes method/model/seed). **205 passed** (was 170); ruff/black clean.
  - `pyproject.toml` — `ui` extra (`streamlit==1.41.1`) + `pythonpath = ["."]` so the
    root-level `app` package imports in tests. The Streamlit entrypoint bootstraps
    the repo root onto `sys.path` so `app.*` package imports resolve under
    `streamlit run` (which otherwise puts only `app/` on the path).
  - **Validated headlessly** via Streamlit's `AppTest`: query *"practical deep
    learning"* → `DATA C182` (Deep Neural Networks) top hit; technique blurbs, the
    leaderboard `family` column, both glossary expanders, and the Map view (base +
    seed-overlay, 3-layer Altair spec over 11,073 points) all render with no
    exceptions in any of the four views. ADR-0012.
- **Phase 7 / B.8c — "why this fits" explainer (closes Track B.8).** The last B.8
  piece, and the one place the two negative results (tag rung ADR-0009, reranker
  ADR-0010) pointed: not ranking, but *justifying* a ranking SBERT already
  produced.
  - `src/courserec/recommenders/llm.py` — `RecommendationExplainer`: given a query
    (a seed course's text or a free-text search) and one already-recommended
    candidate, returns a single short "why this fits" sentence for the Phase 8 UI.
    Deliberately **not** a `Recommender` — no `list[Rec]`, no ranking, and **never
    scored by `eval.py` or the leaderboard** (an explanation has no ground-truth
    ordering to measure; ADR-0011). `fit` captures `text`/`title` maps and opens
    the cache (returns `self`); generation is lazy at `explain` time.
    `explain_seed` resolves the seed's own text for item-to-item mode.
  - `OllamaClient.explain` — (query, candidate title + text) → one validated
    sentence under a JSON-schema `format` (`{"reason": str}`), deterministic
    (`temperature=0`, `seed=RANDOM_SEED`, `think=False`), whitespace-normalized.
    Zero new dependencies (reuses the ADR-0009 stdlib-`urllib` client).
  - `_ExplanationCache` — `sha1(model + normalized-query + candidate-id)` →
    `artifacts/llmcache/<model>/explanations.json`; one deterministic call per
    (query, candidate), ever.
  - **Degrades to `None`, never raises on unavailability** — empty query, blank
    model reason, or Ollama down with no cached entry all return `None` (the UI
    omits the line). Unlike the two ranking rungs, `fit` never skips: an optional
    UI line has no "useless duplicate" failure mode. Only an unknown `candidate_id`
    raises (`KeyError`, a programming bug).
  - `scripts/explain_recs.py` — CLI driver (`--seed`/`--query`, SBERT base, live
    Ollama or a clear exit) to validate the rung and warm the cache for the UI.
  - `tests/test_llm.py` — 10 new tests via `FakeExplainClient` (no daemon): reason
    propagation, seed-text resolution, cache reuse, warm-offline hit, cold-offline
    `None`, empty-query `None`, blank-reason `None`, unknown-candidate `KeyError`,
    before-`fit` `RuntimeError`, bad-config. **170 passed** (was 160).
  - **Validated live (qwen3:8b):** concrete on-topic one-liners in both modes
    (e.g. `COMPSCI 189 → STAT C241A`: "Both courses cover statistical learning
    theory, classification, regression, clustering…"). ADR-0011 written;
    `docs/adr/README.md` updated.

### Measured
- **Phase 7 / B.8b — zero-shot LLM reranker: measured, does not beat the base.**
  Ran `scripts/run_eval.py` with Ollama up (qwen3:8b) to fill `reranks.json` (1117
  reranks cached: 1072 cross-listing seeds + 44 judged queries), then a warm rerun
  to confirm reproducibility.
  - **Cross-listing lens:** NDCG@10 **0.9649** [0.957, 0.972] vs MiniLM base
    **0.9710** [0.965, 0.977] — Δ −0.006, CIs overlap. recall@10 dips 1.000 →
    0.9921 (a twin occasionally reordered out of the top-10).
  - **Judged free-text lens:** NDCG@10 **0.6559** [0.586, 0.729] vs base **0.6821**
    [0.615, 0.746] — Δ −0.026, CIs overlap. NDCG@5 flat (+0.004). recall@10
    0.7056 → 0.6671.
  - **Verdict:** a second documented negative result (joins the tag rung). recall@20
    is identical base↔reranker (pure reorder); SBERT's top-20 is already at/near
    the recall ceiling, so there is no headroom to exploit — the Phase 4
    cross-encoder trap with a zero-shot LLM. **SBERT MiniLM stays the top rung.**
  - **Cost:** ~4.4 s/query (cross-list), ~3.7 s/query (text) on a cold cache —
    ~13000× the base latency. **Reproducibility:** warm rerun is fully offline and
    the metric columns are byte-identical across runs (only timing differs).
  - `results/leaderboard{,_text,_heldout}.{md,csv}` regenerated. ADR-0010 verdict
    written in; `docs/RESULTS.md`, `docs/TRADEOFFS.md`, `README.md` synced.

### Added
- **Phase 7 / B.8b — zero-shot LLM reranker.**
  - `src/courserec/recommenders/llm.py` — `LLMRerankRecommender`: retrieves the
    top-N (default 20) from a base ranker (MiniLM `SbertRecommender` by default,
    any `Recommender` injectable, seed already excluded), then reorders those
    candidates with **one** deterministic Ollama call over their **full** text (no
    distillation — the lesson of ADR-0009). The model returns an integer
    permutation under a JSON-schema `format`; `_reconcile` maps it back to ids,
    drops out-of-range/duplicate picks, and appends any omitted candidate in base
    order, so the output ranks every candidate exactly once however the model
    behaves. Scores are rank-based (`len - position`), strictly descending.
  - `OllamaClient.rank_candidates` — numbered-listing prompt → validated integer
    ranking; reuses the stdlib-`urllib` client (still **zero new dependencies**).
  - `_RerankCache` — reranks cache to `artifacts/llmcache/<model>/reranks.json`
    keyed `sha1(model+query+candidate-ids)`; one LLM call per (query,
    candidate-set), ever.
  - **Graceful degradation:** when Ollama is down the rung falls back to the base
    order; `fit` raises `LLMUnavailable` (skip+flag) only when down *and* the
    rerank cache is cold (it would be a useless base-order duplicate).
  - `scripts/run_eval.py` — `LLMRerankRecommender()` added to the sweep (19
    techniques total) under the existing `LLMUnavailable` graceful-skip `except`.
  - `tests/test_llm.py` — 12 reranker contract tests via `FakeBase` +
    `FakeRerankClient` (no daemon): reorder, seed-exclusion, sort/cap, cache reuse,
    cold-offline skip, warm-offline cache, uncached-offline base fallback,
    malformed-output reconcile, by-text, empty-query, unknown-seed, bad-config.
    Suite now **160 passed**. Live smoke against qwen3:8b confirms the
    prompt/schema/parse round-trip.
  - `docs/adr/0010-llm-reranker.md` — rerank-prompt + caching + fallback design.

### Changed
- **Phase 7 — full-catalog enrichment overturns the LLM tag rung's provisional
  win.** Ran `scripts/enrich_catalog.py --all` (qwen3:8b, 8,535 fresh generations,
  ~5 h, cached/resumable) → **100% catalog coverage** (10,900/11,073 non-empty).
  Re-running the eval de-confounds the rung: cross-listing NDCG@10 0.960 → **0.957**
  (now tied with the lexical cluster, mid-pack) and free-text 0.585 → **0.404**
  (now *below* tfidf 0.461). The subset-run lift was the target/distractor
  vocabulary-separation artifact; distilling a description to ~6–12 tags loses more
  discriminative signal than the LLM's synonym-normalization adds. RESULTS Phase 7
  + ADR-0009 rewritten with the corrected (negative) verdict.
- **`scripts/run_eval.py`** — `_llm_coverage_note` is now coverage-adaptive: at
  ≥95% catalog coverage the boards show an "LLM enrichment (full): comparable" note
  instead of the "Partial LLM enrichment" artifact caveat.

### Added
- **Phase 7 / B.8 — LLM enrichment via local Ollama (tag-extraction rung).**
  - `src/courserec/recommenders/llm.py` — `LLMTagRecommender`: a local LLM
    (Ollama, **qwen3:8b**, no API key) extracts structured tags
    (topics/skills/level/prereqs-mentioned) per course via Ollama's JSON-schema
    `format`; the rung ranks by TF-IDF cosine over the tag **profile**
    (topics+skills+prereqs), with raw-text fallback for un-enriched courses.
    `OllamaClient` is stdlib `urllib` → **zero new dependencies**. `fit` reads the
    tag cache only (never generates); tags cache to `artifacts/llmcache/<model>/`
    keyed `sha1(model+normalized_text)`. Skips+flags (`LLMUnavailable`) only when
    nothing is cached *and* Ollama is unreachable — else degrades to raw text
    (ADR-0009).
  - `scripts/enrich_catalog.py` — the slow, resumable generation pass; enriches
    the **eval-relevant subset** (cross-listing seeds+twins + judged gold) by
    default, `--all` for the full catalog, `--model` to swap models.
  - `scripts/run_eval.py` — `LLMTagRecommender` added; `LLMUnavailable` joins the
    graceful-skip `except`; a **"Partial LLM enrichment"** coverage note is written
    onto all three leaderboards so the confounded number is not misread.
  - `src/courserec/config.py` — `OLLAMA_HOST`, `DEFAULT_LLM_MODEL`.
  - `tests/test_llm.py` — 13 tests with a `FakeClient` (no daemon): contract,
    cache-read `fit`, resumable enrichment, raw-text fallback, skip-when-cold.
    Suite: **148 passed** (was 135).
  - **First result (subset-enriched, ADR-0009 / RESULTS Phase 7):**
    `llm_tags(qwen3:8b)` tops every lexical/topic baseline on both lenses
    (cross-list 0.960 vs tfidf 0.955; free-text 0.585 vs 0.461), behind only SBERT
    — **but only 1,390/11,073 courses (12.5%, the eval targets) are enriched**, so
    part of the lift is a target/distractor vocabulary-separation artifact. The
    free-text win (query enriched live → tag normalization) is the trustworthy
    signal; `enrich_catalog.py --all` is the documented clean-confirmation step.
- **Phase 5 / B.5 — metadata fusion (weighted one-hot facets ⊕ TF-IDF).**
  - `src/courserec/recommenders/metadata.py` — `MetadataRecommender`, a Track-B
    ranker that fuses a TF-IDF text block with a one-hot **subject + department +
    level + units** block under a single weight λ (`text_weight`). Each block is
    L2-normalized so the fused score is `λ²·cos_text + (1−λ)²·cos_meta`; λ=1.0 is
    bit-for-bit the `tfidf` baseline, making the metadata contribution an exact
    ablation. Fully sparse (TF-IDF backend) → no optional extra or API key; absent
    facet columns / null values are skipped (fits a metadata-poor frame).
    `recommend_by_text` zeroes the metadata block (a query has no facets), so the
    text lens reduces to pure TF-IDF (ADR-0008).
  - `scripts/run_eval.py` — λ ∈ {0.9, 0.7, 0.5} sweep wired into
    `build_recommenders`; the three configs land on the cross-listing + judged-text
    leaderboards (regenerated).
  - `tests/test_metadata.py` — contract + fusion-behavior tests (seed excluded,
    sorted/capped, λ=1 ≡ TF-IDF, metadata pulls a same-facet course up, sparse-text
    safe, missing-column safe, λ∉[0,1] rejected, artifact round-trip). Suite:
    **135 passed** (was 125).
  - **Honest finding (ADR-0008, RESULTS Phase 5/B.5):** fusion *loses* on the
    primary cross-listing lens, monotonically in λ (0.948 → 0.936 → 0.909, all
    below the 0.955 TF-IDF baseline), because **99.7% of cross-listing edges span
    different subjects** — the one-hot subject/dept block pushes twins apart. Ties
    the baseline exactly on the free-text lens. Its real value is browse coherence
    (a weak proxy the harness warns not to optimize) + sparse-text robustness.
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
  - ADRs: [0003](https://github.com/sandeep-jay/course-recommender-lab/blob/main/docs/adr/0003-judged-query-lens.md) (judged-query lens),
    [0004](https://github.com/sandeep-jay/course-recommender-lab/blob/main/docs/adr/0004-semantic-vectors.md) (semantic vectors, caching, ANN).
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

