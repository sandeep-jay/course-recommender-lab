# ADR-0012: Phase 8 minimal Streamlit UI

**Date:** 2026-06-20
**Status:** Accepted

## Context
Phases 0–7 built the techniques, the eval harness, the leaderboard, and the Phase
7c "why this fits" explainer — but everything is reachable only from scripts. Plan
§4 asks for a **minimal** UI to surface the work: Explore (course or free-text →
top-k with scores and the optional why-line), Compare (one query, two techniques),
and Leaderboard (`results/leaderboard.csv` + the UMAP map). Explicitly "no auth, no
database, no styling beyond Streamlit defaults." Load-bearing questions:

1. **Where does the UI live, and how is its logic testable?** Streamlit scripts run
   under a script-runner, not a normal import, and are awkward to unit-test.
2. **Which techniques does it expose?** The leaderboard sweeps ~18 technique×config
   rows, several of them slow, key-gated, or download-heavy.
3. **How does it stay fast and offline?** Re-fitting on every widget interaction, or
   hard-failing when Ollama is down, would make it unusable.

## Decision
1. **A root-level `app/` package, split into a testable core and a thin UI.** The
   technique choices and display formatting live in `app/registry.py`, which
   **imports no Streamlit** — so they are unit-tested in the base environment.
   `app/streamlit_app.py` is the only Streamlit-dependent module and is a thin view
   layer over the registry. Streamlit is a new optional extra `ui`
   (`pip install -e ".[ui,semantic]"`); the library and `scripts/run_eval.py` never
   import it. `pyproject` gains `pythonpath = ["."]` so the root-level `app` package
   imports in tests (`courserec` resolves via the editable install).
2. **A curated, offline, no-API-key subset on the leaderboard.** The UI exposes six
   representative rungs — SBERT MiniLM (default), SBERT MPNet, TF-IDF, BM25, LSA,
   Metadata+text — a spread across semantic / lexical / topic / metadata families.
   The default is **SBERT MiniLM**, the rung that tops both ranking lenses
   (RESULTS.md). Heavier or key-gated rungs (API embeddings, the LLM tag/rerank
   rungs, the cross-encoder reranker, the graph model's held-out-edge eval) are left
   to `scripts/run_eval.py` — the UI favours fast, reproducible, offline retrieval,
   and the *full* comparison still lives in the Leaderboard view's table.
3. **Cache fitted instances, never re-fit on interaction.** The catalog, each fitted
   technique (keyed by name), and the explainer are wrapped in
   `st.cache_resource`, so picking a course or flipping a toggle re-runs the script
   but reuses the fitted objects; combined with the `artifacts/` cache the first
   interaction is fast and subsequent ones instant.
4. **The why-line is opt-in and degrades silently.** Explanations are behind an
   off-by-default checkbox (they cost a local LLM call), reuse
   `RecommendationExplainer` (ADR-0011) unchanged, and inherit its degrade-to-`None`
   behaviour — when Ollama is down the column shows `—` and the view adds a caption
   on how to enable it, never an error. A text-incapable technique in query mode
   surfaces an `st.info`, not a crash (none of the six are, but the guard is real).

## Alternatives considered
- **Put the logic inside the Streamlit script.** Rejected: the script-runner makes
  pure-Python logic (which techniques, how to label a course) hard to test. Splitting
  the registry out keeps the contract under unit test and the script a thin shell.
- **Expose every leaderboard row in the picker.** Rejected: API embeddings need a
  key, the LLM rungs and cross-encoder reranker are slow / download-heavy, and the
  graph model is item-only on a held-out split. A six-rung offline subset is the
  honest "explore the techniques" surface; the Leaderboard view still shows them all.
- **A heavier stack (FastAPI + React, or a hosted DB).** Rejected: the plan asks for
  *minimal*, and there is no user-interaction data to persist. Streamlit is one file
  to run and zero infra.
- **Build the catalog map live in the app (UMAP on each load).** Rejected: UMAP over
  11k vectors is slow and stochastic. The Phase 6 diagnostic already writes
  `results/plots/embedding_map.png`; the view just renders it.

## Consequences
**Positive.** Every prior phase is now reachable from one `streamlit run`: the
techniques (Explore/Compare), the explainer (the why-line), and the full eval
(Leaderboard). The testable split means the UI's load-bearing choices have contract
tests (`tests/test_app_registry.py`) that run with no browser and no Streamlit
installed. Caching + `artifacts/` keep it responsive; the whole thing runs offline,
with the LLM why-line as the one opt-in online nicety.

**Validated headlessly (2026-06-20).** Streamlit's `AppTest` drives the real widget
tree: the free-text query *"practical deep learning"* under the default rung returns
`DATA C182` (Designing, Visualizing and Understanding Deep Neural Networks) as the
top hit; Compare renders two side-by-side tables; Leaderboard renders the 18-row
table and the UMAP image — all three views with no exceptions.

**Honest caveats.** (1) The six exposed rungs are a curated subset, not the full
sweep — by design, but a reviewer wanting API/LLM/graph rungs must use
`scripts/run_eval.py`. (2) The why-line needs a running Ollama and is qwen3:8b's
output (ADR-0011's caveats carry over). (3) The seed-course picker lists all ~11k
courses; Streamlit's searchable selectbox handles it, but it is not paginated.

**Neutral.** No leaderboard or eval change — the UI consumes `eval.py` output, it
does not feed it. Streamlit is optional; absent it, the library, tests, and eval are
unaffected.

## Implementation notes
`app/registry.py`: `TECHNIQUE_FACTORIES` (display-name → lazy factory),
`DEFAULT_TECHNIQUE`, `technique_names`, `make_recommender` (`KeyError` on unknown),
`course_label` (`"<id> — <title>"`, falls back to the bare id on a missing title).
`app/streamlit_app.py`: `st.cache_resource`-wrapped catalog / fitted-technique /
explainer loaders, three view functions, and a sidebar-radio dispatcher. `pyproject`
adds the `ui` extra (`streamlit==1.41.1`) and `pythonpath = ["."]`. Tests:
`tests/test_app_registry.py` (default is registered + first, every factory builds a
real `Recommender`, unknown name `KeyError`, label format + missing-title fallback).
Builds on the explainer in [ADR-0011](0011-llm-explainer.md). **Implements plan §4.**

**Addendum — explanatory glossary layer.** Scores and cryptic technique keys
(`sbert(all_minilm_l6_v2,idx=flat)`) mean little without context, so a second
import-safe, Streamlit-free module `app/glossary.py` applies the same
testable-core pattern as the registry: a one-line blurb per exposed technique, a
paragraph per *family* (`family_of`/`family_label` map any raw leaderboard name to a
family, so even rows the UI never fits are explained), a definition per leaderboard
metric (`metric_help`, powering per-column header tooltips), and the three eval
lenses + the leakage guardrail. Descriptions stay honest to the findings (e.g.
metadata fusion *hurting* the cross-listing target is stated). Wired into Explore and
both Compare columns (a picker blurb) and the Leaderboard (a `family` column,
hover-for-definition tooltips, and "How to read this leaderboard" / "Technique
families" expanders). Tested by `tests/test_app_glossary.py` — every exposed
technique has a blurb, every real leaderboard name resolves to a known family, every
metric column is defined. This is the same decision (an import-safe core under unit
test + a thin Streamlit view), not a new one, so it extends this ADR rather than
opening another.
