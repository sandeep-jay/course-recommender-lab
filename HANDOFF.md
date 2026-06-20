# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–8 are green: `pytest` = **205 passed**, `ruff`/`black` clean. **Phase 8
(minimal Streamlit UI, plan §4) is complete** — every prior phase is now reachable
from one `streamlit run app/streamlit_app.py`.

Four views: **Explore** (course or free-text → top-k with scores + an opt-in "why
this fits" column wired to the Phase 7c explainer), **Compare** (one query, two
techniques side by side), **Leaderboard** (the main `leaderboard.csv` table, the
graph technique's **held-out-edge board** `leaderboard_heldout.csv` as a clearly-
labeled "not comparable" second table, and the static Phase 6 map), and **Map** (a
live, interactive Altair projection where a selected seed + its top-k SBERT
recommendations light up). The load-bearing choice
(ADR-0012) is a pattern of **import-safe, Streamlit-free, unit-tested modules feeding
a thin view layer**: `app/registry.py` (technique factories + label helper),
`app/glossary.py` (the **explanatory layer** — per-technique blurbs, per-family
paragraphs, per-metric definitions, the three eval lenses + leakage note), and
`app/projection.py` (cached 2-D layout, keyed by `(method, model, seed)` in
`artifacts/map/`). The glossary is wired in everywhere users need context: a blurb
under each technique picker (Explore + both Compare columns), and on the Leaderboard
a `family` column, hover-for-definition column tooltips, and "How to read this
leaderboard" / "Technique families" expanders. The UI exposes a **curated offline
subset** of six rungs (SBERT MiniLM (default), SBERT MPNet, TF-IDF, BM25, LSA,
Metadata+text); the *full* sweep stays in `scripts/run_eval.py` and the Leaderboard
table. Catalog, fitted techniques, the explainer, and the projection are
`st.cache_resource`-cached so interactions never re-fit; the why-line degrades to
`—` when Ollama is down.

Validated **headlessly** via Streamlit's `AppTest` (query "practical deep learning"
→ `DATA C182` top hit; technique blurbs, the leaderboard `family` column, both
glossary expanders, and the Map view base + seed-overlay all render; no exceptions in
any of the four views). New optional extra `ui` (`streamlit==1.41.1`) + `pythonpath =
["."]` in `pyproject`; the entrypoint bootstraps the repo root onto `sys.path` so
`app.*` imports resolve under `streamlit run`. **Map note:** UMAP needs the `viz`
extra; without it the projection falls back to t-SNE (~12 s cold over ~11k vectors,
then cached, instant warm). ADR-0012 (+ glossary & Map addenda) written; `README.md`,
CHANGELOG, ADR index all synced.

**SBERT MiniLM remains the top rung on both ranking lenses; the UI defaults to it.**

## Next task

Open — the roadmap's planned phases (0–8) are complete. Candidate follow-ups, all
optional: (a) **deploy the UI** (Streamlit Community Cloud / a Dockerfile) so the
portfolio piece is clickable; (b) a **reranker follow-up** sweep (TF-IDF base for
real reorder headroom, or qwen3:32b) — likely low ROI, the rung is closed; (c) a
**`docs/RESULTS.md` Phase 8 note** if a deeper writeup is wanted (the UI itself is
not a scored technique, so RESULTS was not changed this session).

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| What's next now Phases 0–8 are complete | Deploy the UI (clickable portfolio piece) / reranker follow-up (low ROI) / call the roadmap done and polish docs | Sandeep | next session |

## Blockers / waiting-on

None. The repo runs end-to-end offline; Ollama is only needed at query time to
generate a *fresh* tag / rerank / explanation (all caches degrade gracefully or
serve warm). The UI needs `pip install -e ".[ui,semantic]"`.

## First task for next session

Decide the Open-decisions item. If undecided, **deploy the UI** is the highest-ROI
portfolio move: it makes Phases 0–8 a single clickable demo. The app is
`app/streamlit_app.py` (run with `streamlit run app/streamlit_app.py`).
