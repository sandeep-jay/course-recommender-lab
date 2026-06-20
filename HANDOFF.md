# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–8 are green: `pytest` = **184 passed**, `ruff`/`black` clean. **Phase 8
(minimal Streamlit UI, plan §4) is complete** — every prior phase is now reachable
from one `streamlit run app/streamlit_app.py`.

Three views: **Explore** (course or free-text → top-k with scores + an opt-in "why
this fits" column wired to the Phase 7c explainer), **Compare** (one query, two
techniques side by side), **Leaderboard** (`results/leaderboard.csv` table + the
Phase 6 UMAP map). The load-bearing choice (ADR-0012) is a **two-file split**: an
import-safe, Streamlit-free `app/registry.py` (technique factories + label helper,
**unit-tested**) feeds a thin view layer `app/streamlit_app.py`. The UI exposes a
**curated offline subset** of six rungs (SBERT MiniLM (default), SBERT MPNet,
TF-IDF, BM25, LSA, Metadata+text) — fast, no API key, no model downloads beyond the
SBERT cache; the *full* sweep (API/LLM/cross-encoder/graph rungs) stays in
`scripts/run_eval.py` and the Leaderboard table. Catalog, fitted techniques, and the
explainer are `st.cache_resource`-cached so interactions never re-fit; the why-line
degrades to `—` when Ollama is down.

Validated **headlessly** via Streamlit's `AppTest` (query "practical deep learning"
→ `DATA C182` top hit; all three views render with no exceptions). New optional
extra `ui` (`streamlit==1.41.1`) + `pythonpath = ["."]` in `pyproject`. ADR-0012
written; `README.md`, CHANGELOG, ADR index all synced.

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
