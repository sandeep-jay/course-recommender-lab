# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–7c are green: `pytest` = **170 passed**, `ruff`/`black` clean. **Track B.8
(LLM enrichment) is complete** — tags → reranker → explainer. The new **Phase 7c
"why this fits" explainer** (`RecommendationExplainer`, `recommenders/llm.py`) is
the last B.8 piece and the one the two negative ranking results (tags ADR-0009,
reranker ADR-0010) pointed to: spend the LLM **justifying** an SBERT ranking, not
producing one.

By design it is **not a `Recommender` and not on the leaderboard** — a free-text
justification has no ground-truth ordering to score (ADR-0011). It returns one
sentence per (query, candidate) pair via one deterministic qwen3:8b call under a
`{"reason": str}` schema, cached by `sha1(model+query+candidate-id)` in
`explanations.json`. Because the "why" line is *optional*, every unavailability path
degrades to `None` (UI omits the line); `fit` never skips. Validated **live**
(qwen3:8b) via `scripts/explain_recs.py` — concrete on-topic one-liners in both
item-to-item and free-text modes; cache persists and repeat pairs serve with no
call. ADR-0011 written; `RESULTS.md`, `TRADEOFFS.md`, `README.md`, CHANGELOG, ADR
index all synced.

**SBERT MiniLM remains the top rung on both ranking lenses; the LLM's earned role is
explanation, not ranking.**

## Next task

**Phase 8 Streamlit UI** (recommended) — it surfaces everything built: the
`Recommender` leaderboard, item-to-item + free-text modes, and the new Phase 7c
"why this fits" line (already wired and live). Alternative: a **reranker follow-up**
sweep (TF-IDF base for real reorder headroom, or qwen3:32b) — likely low ROI given
the near-ceiling SBERT base, and the reranker rung is closed.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| What's next now Track B.8 is complete | Phase 8 Streamlit UI (recommended — surfaces the explainer + leaderboard) / reranker follow-up (TF-IDF base or qwen3:32b — likely low ROI) | Sandeep | next session |

## Blockers / waiting-on

None. The repo runs end-to-end offline; Ollama is only needed at query time to
generate a *fresh* tag / rerank / explanation (all caches degrade gracefully or
serve warm).

## First task for next session

Decide the Open-decisions item, then start it. Recommended: **Phase 8 Streamlit UI**
— three views per plan §4 (Explore with the "why this fits" line, Compare two
techniques, Leaderboard + UMAP map). The explainer (`RecommendationExplainer`) is
ready to drop in via `explain_seed` / `explain` on top of the SBERT base.
