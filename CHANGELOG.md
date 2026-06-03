# Changelog

All notable changes to course recommender implementation
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
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

