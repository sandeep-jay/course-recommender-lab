# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

The roadmap's planned phases **0–8 are all complete and green**: `pytest` = **205
passed**, `ruff`/`black` clean. Phase 8 is the four-view Streamlit UI
(`streamlit run app/streamlit_app.py`) over import-safe, unit-tested modules
(`app/registry.py`, `app/glossary.py`, `app/projection.py`); SBERT MiniLM is still
the top rung and the UI default. No roadmap work is pending — the one open thread is
*what to do next* (deploy vs. wind down), an owner decision below.

## Next task

Decide the open-decisions row, then start it. If deploying the UI: add a `Dockerfile`
(or a Streamlit Community Cloud config) that runs `streamlit run app/streamlit_app.py`
with `pip install -e ".[ui,semantic]"`, so Phases 0–8 become one clickable demo.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| What's next now Phases 0–8 are complete | Deploy the UI (clickable portfolio piece — recommended) / reranker follow-up (TF-IDF base or qwen3:32b — low ROI, rung closed) / call the roadmap done and polish docs | Sandeep | next session |

## Blockers / waiting-on

None. The repo runs end-to-end offline; Ollama is only needed at query time for a
*fresh* tag/rerank/explanation (caches degrade gracefully or serve warm). The UI
needs `pip install -e ".[ui,semantic]"` (add `viz` for the Map's faster UMAP).

## First task for next session

Decide the open-decisions row; if undecided, scaffold a `Dockerfile` that serves
`app/streamlit_app.py` to make the UI deployable.
