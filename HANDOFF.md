# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–7a are green: metadata fusion (Phase 5/B.5) and the **LLM tag-extraction
rung (Phase 7/B.8, part a)** both land on the leaderboards, with `pytest` = **148
passed** and `ruff`/`black` clean. The LLM rung
(`src/courserec/recommenders/llm.py`, `LLMTagRecommender`) ranks by TF-IDF cosine
over **local-Ollama** (qwen3:8b, no key, stdlib `urllib`, zero new deps) tag
profiles; `fit` only reads a cache that the resumable `scripts/enrich_catalog.py`
pass fills. First result: it tops every lexical baseline on both lenses (cross-list
0.960, free-text 0.585) **but only 1,390/11,073 courses (12.5%, the eval targets)
are enriched**, so the cross-listing number is confounded by target/distractor
vocabulary separation (leaderboards carry a "Partial LLM enrichment" note;
ADR-0009). The free-text win (query enriched live → tag normalization) is the
trustworthy signal.

## Next task

Two open Phase-7 threads — pick either:
1. **Confirm the LLM rung cleanly:** run `python scripts/enrich_catalog.py --all`
   (full ~11k catalog, multi-hour, resumable, cached) then re-run `run_eval`; this
   removes the partial-enrichment confound and gives a number comparable to the
   other rungs.
2. **Build the rest of Phase 7** (plan §2.8 b/c): the **zero-shot LLM reranker**
   over candidate sets and the **"why this fits"** explanation for the UI, plus the
   **LLM-as-judge** validation of the free-text lens (plan §3). Reuse the existing
   `OllamaClient` + tag cache.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Next Phase-7 step | Full-catalog enrichment to de-confound the tag rung / build the zero-shot reranker + "why this fits" / start Phase 8 Streamlit UI | Sandeep | next session |

## Blockers / waiting-on

None. (LLM rung needs `ollama serve` up + `qwen3:8b` pulled for enrichment; eval
runs fine offline against the warm cache.)

## First task for next session

Decide between full-catalog enrichment (`enrich_catalog.py --all`, the clean
confirmation) and building the zero-shot reranker. If the reranker: add a
`LLMRerankRecommender` that retrieves top-N with SBERT/TF-IDF then reorders with a
single Ollama call (cached, deterministic), reusing `OllamaClient`; degrade
gracefully when Ollama is down; contract test with the `FakeClient` pattern; wire
into `run_eval`; ADR for the rerank-prompt + caching design.
