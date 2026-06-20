# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–7b are green: `pytest` = **160 passed**, `ruff`/`black` clean. The Phase 7
**zero-shot LLM reranker** (`LLMRerankRecommender`) is now **measured** and settled
as a second documented negative result (joining the tag rung, ADR-0009): it **does
not beat the SBERT MiniLM base** on either ranking lens.

| Lens | base NDCG@10 (CI) | reranker NDCG@10 (CI) | Δ | recall@10 |
|---|---|---|---|---|
| Cross-listing (1072) | 0.9710 [0.965, 0.977] | 0.9649 [0.957, 0.972] | −0.006 | 1.000 → 0.992 |
| Judged free-text (44) | 0.6821 [0.615, 0.746] | 0.6559 [0.586, 0.729] | −0.026 | 0.706 → 0.667 |

Both deltas are negative and inside the bootstrap CIs (no significant difference);
recall@10 actually dips. recall@20 is identical base↔reranker (pure reorder) —
SBERT's top-20 is already near the recall ceiling, so reordering has no headroom and
only room to hurt (the Phase 4 cross-encoder trap with a zero-shot LLM). Cost ~4 s/
query on a cold cache (~13000× the base). `reranks.json` is now **warm** (1117
entries); the suite reruns fully offline and the metric columns are byte-identical
across cold↔warm runs (reproducibility confirmed). Verdict written into **ADR-0010**;
`RESULTS.md`, `TRADEOFFS.md`, `README.md`, leaderboards all synced.

**SBERT MiniLM remains the top rung on both ranking lenses.**

## Next task

Pick one (see Open decisions): **Phase 7c "why this fits"** explanation (the last
B.8 part — an LLM justification over full candidate text, reusing the warm Ollama
client + tag/rerank caches), **Phase 8 Streamlit UI**, or a **reranker follow-up**
sweep (TF-IDF base for real reorder headroom, or qwen3:32b). The reranker rung itself
is closed — no further measurement needed.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| What's next now the reranker is settled | Phase 7c "why this fits" explanations (finish B.8) / Phase 8 Streamlit UI / reranker follow-up (TF-IDF base or qwen3:32b — likely low ROI given the near-ceiling base) | Sandeep | next session |

## Blockers / waiting-on

None. (`reranks.json` is warm, so `run_eval.py` reruns fully offline; Ollama is only
needed at query time on a *cold* cache or for the future "why this fits" rung.)

## First task for next session

Decide the Open-decisions item, then start it. Recommended: **Phase 7c "why this
fits"** — it completes Track B.8 and is the one rung that uses the LLM where the
evidence says its value lives (full candidate text, a justification rather than a
ranking), and it reuses the already-warm Ollama client and caches.
