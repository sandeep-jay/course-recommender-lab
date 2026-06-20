# Runbook — course-rec-lab

How to run **everything** in this repo end-to-end: install tiers, the data pipeline,
every script and its flags, every model (and how to get it), the Streamlit UI, the
dev loop, where artifacts live, and how to recover from the common failure modes.

For *why* each piece works the way it does, see the [ADRs](adr/README.md),
[RESULTS.md](RESULTS.md), and [TRADEOFFS.md](TRADEOFFS.md). For the contract every
technique obeys, see [roadmap/recommender_plan.md](roadmap/recommender_plan.md).

---

## 0. Prerequisites

- **Python 3.11+** (the package requires `>=3.11`).
- **Apple Silicon note (this machine):** PyTorch uses the **MPS** backend; keep
  `fp16=False` (the recommenders already do). No CUDA needed.
- **No API key is required** to run the repo through Phase 6. The LLM phases use a
  **local** Ollama daemon (no key, no cost); the one API rung (OpenAI embeddings) is
  optional and skips cleanly when no key is set.
- Everything is **reproducible**: the global seed is `RANDOM_SEED = 42`.

---

## 1. Install tiers

The base install runs Phases 0–2 (data + lexical + topic models). Heavy or
phase-specific dependencies are **optional extras** — install only what you need.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # base + pytest/ruff/black — Phases 0–2 + tests
```

| Extra | Pulls in | Unlocks | Install |
|---|---|---|---|
| `dev` | pytest, ruff, black | tests + lint/format | `pip install -e ".[dev]"` |
| `semantic` | sentence-transformers, torch, faiss-cpu | Phase 3 SBERT rungs, the cross-encoder reranker, clustering, **the UI's semantic views** | `pip install -e ".[semantic]"` |
| `viz` | matplotlib, umap-learn | Phase 6 map figure; the UI Map view's **faster UMAP** projector (else t-SNE) | `pip install -e ".[viz]"` |
| `ui` | streamlit | the Phase 8 app | `pip install -e ".[ui]"` |

**Recommended combos**
```bash
pip install -e ".[dev,semantic]"          # everything except the map figure + UI
pip install -e ".[dev,semantic,viz,ui]"   # the lot (run the full UI incl. fast Map)
```

Techniques **degrade gracefully** when an extra is absent: without `semantic`, the
SBERT/API/rerank rungs skip and note it (they never hard-fail the suite); without
`umap-learn`, projections fall back to scikit-learn t-SNE.

---

## 2. Data setup

The raw catalog CSV is **gitignored** (keep your own copy) and expected at
`data/raw/courses-report_2026-06-02.csv` (`config.RAW_CATALOG_CSV`). Build the
processed parquet once:

```bash
python scripts/prepare_data.py     # raw CSV -> data/processed/courses.parquet
```

This loads with pandas (RFC-4180 quoted newlines), replaces the catalog's `"-"`
null token with real NA, synthesizes `course_id = f"{Subject} {Course Number}"`,
collapses duplicate ids, and builds the `text` field (title + description, with a
title fallback for sparse rows). Every later step reads
`data/processed/courses.parquet` (`config.load_processed()`).

The hand-labeled free-text ground truth lives at `data/judged_queries.json` (force-
committed via a `.gitignore` exception — it is curated, not regenerated). Inspect it:

```bash
python scripts/build_judged_queries.py --stats              # summarize the set
python scripts/build_judged_queries.py --suggest "deep learning"   # lexical candidates
```

---

## 3. The pipeline, script by script

All scripts are regenerable in one command and write under `results/` or
`artifacts/`. Order below is the natural full run.

### 3a. Evaluate + leaderboards — `scripts/run_eval.py`
Fits **every** configured technique, scores it on all three lenses, and writes the
boards. The single source of truth for rankings.

```bash
python scripts/run_eval.py                  # full sweep (uses bootstrap=1000)
python scripts/run_eval.py --bootstrap 100  # faster CI estimate (lower is quicker)
```
**Writes** (each `.md` + `.csv`):
- `results/leaderboard.*` — cross-listing lens, sorted by NDCG@10 (the default).
- `results/leaderboard_text.*` — the judged free-text lens (`recommend_by_text`).
- `results/leaderboard_heldout.*` — the held-out cross-listing-edge split (the only
  split the **graph** technique may be scored on; the whole sweep is re-scored here
  for a fair comparison).

LLM rungs are scored from a **warm cache** here (see 3c) so this stays fast; if the
cache is cold and Ollama is down they skip and note it. SBERT artifacts load from
`artifacts/` rather than re-encoding (first ever run encodes ~11k docs, ~10s).

### 3b. Clustering diagnostic + 2-D map — `scripts/run_clustering.py`
Phase 6. Clusters the cached SBERT embeddings (KMeans / agglomerative / HDBSCAN) and
projects them to 2-D. A **diagnostic, not a recommender** — never on the leaderboard.

```bash
python scripts/run_clustering.py                       # default: 100 clusters, UMAP/t-SNE map
python scripts/run_clustering.py --no-plot             # table only, skip the slow projection
python scripts/run_clustering.py --projection tsne     # force t-SNE (default: auto)
python scripts/run_clustering.py --n-clusters 150 --min-cluster-size 20
```
**Writes:** `results/cluster_report.{md,csv}` and `results/plots/embedding_map.png`.
Needs `semantic` (embeddings); the figure needs `viz` (matplotlib; UMAP optional).

### 3c. LLM tag enrichment — `scripts/enrich_catalog.py`
Phase 7. The slow, explicit pass that populates the LLM **tag cache** via local
Ollama, split out so `run_eval.py` stays fast. Requires a running Ollama (see §5).

```bash
python scripts/enrich_catalog.py                 # enrich the eval subset (default)
python scripts/enrich_catalog.py --all           # enrich the WHOLE catalog (~11k, slow)
python scripts/enrich_catalog.py --model qwen3:32b
```
Writes the tag cache under `artifacts/llmcache/<model>/`. Re-running serves warm
entries with no new calls.

### 3d. "Why this fits" explanations — `scripts/explain_recs.py`
Phase 7c. Drives the `RecommendationExplainer` over the SBERT base — validates the
rung live and warms the explanation cache for the UI. Requires Ollama (§5).

```bash
python scripts/explain_recs.py --seed "COMPSCI 189"            # item-to-item mode
python scripts/explain_recs.py --query "practical deep learning"   # free-text mode
python scripts/explain_recs.py --seed "COMPSCI 189" --k 10 --model qwen3:32b
```
Writes to `artifacts/llmcache/<model>/explanations.json`. Prints a clear message and
exits if Ollama is unreachable (the rest of the repo still runs without it).

---

## 4. The Streamlit UI

```bash
pip install -e ".[ui,semantic]"        # add viz for the Map's faster UMAP
streamlit run app/streamlit_app.py     # run from the repo root
```
Open <http://localhost:8501>. Four views:

- **Explore** — a seed course or free-text query × a technique → top-k with scores,
  plus an **opt-in** "why this fits" column (needs Ollama; degrades to `—` if absent).
- **Compare** — one query through two techniques side by side.
- **Leaderboard** — the main board + the graph's held-out board (clearly labeled "not
  comparable"), with a `family` column and hover-for-definition column tooltips, plus
  the static Phase 6 map.
- **Map** — a live, interactive 2-D projection; pick a seed and its top-k SBERT
  recommendations light up. UMAP/t-SNE toggle; the projection is cached to
  `artifacts/map/` (≈12 s cold on t-SNE over ~11k vectors, instant warm).

> **Must run from the repo root.** `streamlit run` puts `app/` on `sys.path`; the
> entrypoint bootstraps the repo root on so the `app.*` package imports resolve.

### 4a. Deploy the UI (Docker)

A `Dockerfile` packages the UI as a **warm, offline, CPU-only** image (ADR-0013):
the processed catalog, the leaderboard, the six UI rungs' artifacts + the Map
projection, and the default MiniLM weights are all baked in, so the container starts
instantly with no first-load encode and no network.

```bash
docker build -t course-rec-ui .                 # from the repo root
docker run --rm -p 8501:8501 course-rec-ui      # then open http://localhost:8501
```

> **Prerequisite — a warm repo.** The build `COPY`s `data/processed/courses.parquet`
> and `artifacts/`, which are **gitignored** (not in a fresh clone). Build from a repo
> that has already run the pipeline + eval (§2–3), so those exist on disk. The
> `.dockerignore` then ships only the runtime subset (~150 MB of artifacts; the raw
> CSV, regenerable caches, and unused rungs are excluded).

What works offline in the container: course-to-course, Compare, Map, the Leaderboard,
and **free-text queries** (MiniLM is pre-pulled). What still needs network/Ollama on
demand: the **SBERT MPNet** rung (its weights download on first use) and the opt-in
"why this fits" line (needs an Ollama daemon — not reachable from inside the container
unless you point `OLLAMA_HOST` at a host-side daemon).

**PaaS hosts.** The image is `$PORT`-aware and runs headless as a non-root user, so
Cloud Run / HF Spaces (Docker) work unchanged — they inject `$PORT` and the `CMD`
binds it (defaults to 8501).

---

## 4b. Teaching notebooks (`notebooks/`)

Ten step-by-step notebooks (ADR-0014) that build each technique from primitives on
the real catalog and run its eval live — the teaching companion to the library. One
per technique family (`01`–`08`) plus a data/eval foundation (`00`) and a
cross-technique synthesis (`09`).

```bash
pip install -e ".[notebooks,semantic]"          # tooling + the semantic rungs

# Interactive: pair the .py source <-> .ipynb, open in Jupyter
jupytext --to ipynb notebooks/01_lexical.py
jupyter lab notebooks/01_lexical.ipynb

# Headless / CI: generate, then execute every notebook (fails on any cell error)
jupytext --to ipynb notebooks/*.py
pytest --nbmake --nbmake-timeout=600 notebooks/*.ipynb
```

> The **`.py` percent scripts are the versioned source** (clean diffs); the `.ipynb`
> are generated by jupytext and **gitignored** — never edit or commit them. nbmake
> executes the generated notebooks so they can't silently rot; it's an **opt-in**
> lane (`--nbmake`), kept out of the default `pytest` run so the unit suite stays
> fast. Notebooks load fitted models from `artifacts/` (warm) and need the
> `semantic` extra for `03`/`04`/`07`/`08` and a running Ollama for the live LLM
> cells in `08` (which degrade gracefully when it's down). See
> [notebooks/README.md](../notebooks/README.md) for the index.

---

## 5. Ollama (the local LLM, no API key)

The Phase 7 / 7c rungs (tags, reranker, explainer) call a **local** Ollama daemon —
no key, no cost, no network beyond `localhost:11434`. Everything else runs without it.

```bash
# 1. Install Ollama (https://ollama.com), then start the daemon:
ollama serve
# 2. Pull the default model (and optionally the larger one):
ollama pull qwen3:8b
ollama pull qwen3:32b          # optional, used via --model
```
- Default model: `qwen3:8b` (`config.DEFAULT_LLM_MODEL`).
- Override the host with `OLLAMA_HOST` (default `http://localhost:11434`).
- **Graceful degradation:** if the daemon is down, the tag/rerank rungs skip with a
  note, the explainer returns `None` (UI omits the line), and caches still serve warm
  entries — nothing hard-fails.

---

## 6. Models reference

| Model | Used by | Source / how to get | Needs |
|---|---|---|---|
| `all-MiniLM-L6-v2` | SBERT (default, top rung) + the Map/cluster projections | Hugging Face — auto-downloaded on first use | `semantic` |
| `all-mpnet-base-v2` | SBERT (larger variant) | Hugging Face — auto-downloaded | `semantic` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | the rerank rung | Hugging Face — auto-downloaded | `semantic` |
| `text-embedding-3-small` | the **optional** API embedding rung | OpenAI API | `openai` SDK + `OPENAI_API_KEY` env var |
| `qwen3:8b` | LLM tags / reranker / explainer (default) | `ollama pull qwen3:8b` | Ollama daemon |
| `qwen3:32b` | the same rungs via `--model` (optional) | `ollama pull qwen3:32b` | Ollama daemon |

Hugging Face models download once and cache in `~/.cache/huggingface/`. The API rung
reads `OPENAI_API_KEY` from the environment **only** — never from a file, never
hardcoded — and skips when unset.

---

## 7. Dev loop

```bash
pytest                  # full suite (currently 205 tests)
pytest tests/test_app_registry.py -q     # one module
ruff check .            # lint
black .                 # format
```
Tests run with no API key and no Ollama (LLM tests use fakes); the SBERT tests skip
cleanly when `semantic` is absent.

---

## 8. Artifacts & caches

Everything under `artifacts/` is **gitignored** and rebuildable. Delete a subfolder
to force a recompute.

| Path | Holds | Rebuilt by |
|---|---|---|
| `artifacts/<technique>/` | fitted vectors / indexes / SBERT embedding caches | the technique's `fit` (on next run) |
| `artifacts/llmcache/<model>/` | LLM tag, rerank, and explanation caches | `enrich_catalog.py` / `explain_recs.py` / first live UI call |
| `artifacts/map/` | cached 2-D projections (`coords_<method>_<model>_seed42.npy`) | the UI Map view on next load |

The embedding cache key is `sha1(model_name + normalized_text)`; the projection cache
key is `(method, model, seed)`. A shape mismatch triggers a recompute rather than
serving a stale layout.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | ran `streamlit run` from a subdirectory | run from the **repo root** |
| SBERT / rerank / API rungs missing from the leaderboard | `semantic` extra not installed (or no `OPENAI_API_KEY` for the API rung) | `pip install -e ".[semantic]"`; set the key for the API rung |
| Map "UMAP (fast)" actually uses t-SNE | `umap-learn` not installed | `pip install -e ".[viz]"` (the caption always names the projector used) |
| LLM tags/rerank skipped; "why this fits" shows `—` | Ollama daemon down or model not pulled | `ollama serve` + `ollama pull qwen3:8b` |
| First UI interaction / `run_eval` is slow | cold caches (encoding ~11k embeddings, or a cold projection) | one-time; subsequent runs load from `artifacts/` |
| `run_eval` is slow on the CI step | bootstrap resamples | `--bootstrap 100` for a quick pass |
| Leaderboard looks stale | a technique or the eval harness changed | re-run `python scripts/run_eval.py` (never hand-edit the boards) |
| `docker build` fails on `COPY data/...` or `artifacts/...` | building from a cold clone — those are gitignored | run the pipeline + eval first (§2–3) so the warm files exist on disk |
| Docker UI: MPNet rung slow / why-line shows `—` | MPNet weights download on first use; Ollama isn't reachable from the container | expected — MiniLM is offline; point `OLLAMA_HOST` at a host daemon for the why-line |
| Notebook: `No module named 'nbtools'` in an editor | static analysis doesn't know the notebook's cwd | harmless — `nbtools.py` resolves at runtime (the notebook runs from `notebooks/`); nbmake confirms it |
| `pytest --nbmake` times out on `04`/`08` | cross-encoder / LLM cells on CPU | raise `--nbmake-timeout`; they sample seeds to stay fast, but CPU-only is slower than MPS |
