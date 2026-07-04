# ADR-0013: Deploy the UI as a warm, offline Docker image

**Date:** 2026-06-20
**Status:** Accepted

## Context
Phases 0–8 are complete and the Phase 8 Streamlit UI (ADR-0012) runs locally with
`streamlit run app/streamlit_app.py`. To make it a clickable portfolio piece it
needs to run on a fresh host. Two facts from the codebase shape the packaging:

1. **`artifacts/` is gitignored and the catalog is not in git.** A fresh clone has
   no fitted vectors, no FAISS indexes, no Map projection, and no
   `data/processed/courses.parquet`. Re-deriving them on a cold host means encoding
   ~11k SBERT vectors and a t-SNE projection on first load (~30 s, per HANDOFF) —
   and re-running the whole data pipeline.
2. **What each UI path needs at runtime differs.** `recommend_similar`, Compare, and
   the Map load straight from the baked `artifacts/` (no model weights — `fit` calls
   `_load`, never `_load_model`). Only a **free-text query** encodes the query string,
   which needs the MiniLM weights. Sentence-Transformers pulls those from Hugging
   Face on first use — a network dependency at *runtime* unless pre-pulled.
3. **`torch==2.12.0`'s default Linux wheel is the CUDA build (~2 GB+).** A CPU-only
   deploy that pulls it blindly produces a needlessly huge image.

## Decision
Ship a **single CPU-only Docker image that starts warm and needs no network at
runtime**, via a `Dockerfile` + `.dockerignore` + `.streamlit/config.toml`:

1. **Bake the warm state into the image.** `COPY` the processed parquet, `results/`,
   and only the `artifacts/` that back the six UI rungs + the Map. The `.dockerignore`
   drops the raw CSV, the regenerable embedding/LLM caches (`embcache`, `llmcache`,
   `llm_tags`), and the rungs the UI never fits (graph, NMF, LDA, the extra
   TF-IDF/BM25/metadata variants) — ~148 MB of artifacts in, ~100 MB excluded.
2. **Pre-pull the default MiniLM weights at build time** (one `RUN` that constructs
   `SentenceTransformer('all-MiniLM-L6-v2')` into a baked `HF_HOME`), so free-text
   queries work fully offline. The MPNet rung is the deliberate exception: its
   weights download on first use (graceful — slower, needs network), not baked, to
   keep the image lean.
3. **Force the CPU torch wheel.** Install `torch==2.12.0` from
   `https://download.pytorch.org/whl/cpu` *before* `pip install -e ".[ui,semantic]"`,
   so the package install sees torch satisfied and never pulls the CUDA default.
4. **Honor `$PORT`, run headless as non-root.** `config.toml` sets headless / bind
   `0.0.0.0` / usage-stats off; the `CMD` binds `--server.port=${PORT:-8501}` so PaaS
   hosts (Cloud Run, HF Spaces) that inject `$PORT` work unchanged. A non-root
   `appuser` owns the app tree and HF cache; a `/_stcore/health` HEALTHCHECK probes
   readiness.

## Alternatives considered
- **Cold image (no baked artifacts), compute on first load.** Rejected: a ~30 s
  cold-start on every fresh container (and a full data-pipeline run to even produce
  the parquet) is a poor portfolio first impression, and the pipeline's raw CSV
  isn't in the image. Warm trades image size for instant, offline startup.
- **Bake *all* artifacts (the full 250 MB sweep).** Rejected: the UI fits only six
  rungs; shipping NMF/LDA/graph/duplicate variants and the regenerable caches bloats
  the image for paths the UI never exercises. The full sweep stays a `run_eval`
  concern (ADR-0012's curated-subset decision carries over).
- **Pre-pull every model (incl. MPNet, cross-encoder).** Rejected: MPNet's weights
  roughly double the model footprint for a non-default rung. Baking only the default
  keeps the common path offline and the image small; MPNet degrades to a one-time
  download, consistent with the repo's graceful-degradation rule.
- **Default the base PyPI torch wheel.** Rejected: it's the CUDA build on Linux —
  gigabytes of unused CUDA libraries for a CPU deploy.
- **A Streamlit Community Cloud / HF Spaces config instead of Docker.** Not chosen as
  the primary: Docker is the portable, host-agnostic artifact and the `$PORT` +
  headless config already make it Spaces/Cloud-Run-friendly. A Spaces `Dockerfile`
  Space can reuse this image as-is.

## Consequences
**Positive.** `docker build -t course-recommender-lab . && docker run -p 8501:8501
course-recommender-lab` yields a UI that starts warm and offline: course-to-course, Compare,
Map, the Leaderboard, and free-text queries all work with no network and no
first-load encode. CPU-only and `$PORT`-aware, so it runs on a laptop or a PaaS host
unchanged. No application code changed — this is pure packaging over ADR-0012.

**Honest caveats.** (1) The build is **not hermetic**: it `COPY`s `data/processed/`
and `artifacts/`, which are gitignored, so the build host must have a warm repo (a
normal local state after running the pipeline + eval). A truly fresh clone must
regenerate those first — documented in the RUNBOOK. (2) The MPNet rung and the
Phase 7 why-line still need network/Ollama on demand (unchanged degradation). (3)
The image pins `torch==2.12.0` from the CPU index; a version the CPU index lacks
would fail the build loudly (build-time, not runtime).

**Neutral.** No change to the library, eval, leaderboard, or tests; absent Docker,
everything runs exactly as before. Builds on the UI in [ADR-0012](0012-streamlit-ui.md).

## Implementation notes
`Dockerfile` (python:3.11-slim, libgomp1, CPU torch, `.[ui,semantic]`, MiniLM
pre-pull, non-root `appuser`, `$PORT`-aware `CMD`, `/_stcore/health` HEALTHCHECK),
`.dockerignore` (warm-subset allowlist by exclusion), `.streamlit/config.toml`
(headless / bind / usage-stats / no file-watcher). RUNBOOK §"Deploy" documents the
warm-repo prerequisite and the build/run commands. **Implements the HANDOFF deploy
task.**
