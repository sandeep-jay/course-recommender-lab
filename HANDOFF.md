# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–7b are green with `pytest` = **160 passed**, `ruff`/`black` clean. The
Phase 7 **zero-shot LLM reranker** (`LLMRerankRecommender`, `recommenders/llm.py`)
is now built, tested, and wired into the sweep (19 techniques): it retrieves top-N
(default 20) from a MiniLM SBERT base, then reorders those candidates with one
deterministic Ollama call over their **full** text (no distillation — the lesson
of ADR-0009), reusing `OllamaClient` (new `rank_candidates`). The model returns an
integer permutation under a JSON-schema `format`; `_reconcile` always yields a full
permutation however the model behaves. Reranks cache to `reranks.json` keyed
`sha1(model+query+candidate-ids)`; offline it falls back to the base order, and
`fit` skips (`LLMUnavailable`) only when Ollama is down *and* the cache is cold.
Design in **ADR-0010**. The mechanism is **live-verified** against qwen3:8b
(sensible toy reranking), but the **leaderboard delta is not yet measured** — the
full eval was not run this session (it is one LLM call per cross-listing seed +
judged query, ~hour-long on a cold cache).

The prior tag rung (`LLMTagRecommender`) stays settled as not competitive
(ADR-0009); SBERT MiniLM still tops both ranking lenses.

## Next task

**Measure the reranker.** Run `python scripts/run_eval.py` with `ollama serve` up
+ qwen3:8b (warms `reranks.json`; the SBERT artifact is already cached). Read the
delta vs the `sbert(all_minilm...)` base on both lenses — the reranker's ceiling is
the base's recall@20, so it can only reorder what SBERT retrieves. Record the
verdict in ADR-0010 (replace its "unmeasured at write time" caveat) and on the
leaderboards. A second cold run should be free (cache warm) — confirm reproducibility.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| After measuring the reranker | "Why this fits" explanations + LLM-as-judge (finish Phase 7 b/c) / start Phase 8 Streamlit UI / tune the reranker (`retrieve_n`, `candidate_chars`, prompt, qwen3:32b) | Sandeep | next session |

## Blockers / waiting-on

None. (The reranker eval needs `ollama serve` up + `qwen3:8b` at query time; once
`reranks.json` is warm the suite reruns fully offline.)

## First task for next session

Run `python scripts/run_eval.py` (Ollama up, qwen3:8b) to fill `reranks.json` and
get the reranker's NDCG@10 on the cross-listing + judged free-text lenses; compare
against the SBERT base, then write the verdict into ADR-0010 and confirm a warm
rerun is deterministic.
