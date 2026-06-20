# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–7a are green with `pytest` = **148 passed**, `ruff`/`black` clean. The
catalog is now **100% LLM-enriched** (qwen3:8b tags cached for 10,900/11,073
courses) and the de-confounded result is settled: with the full catalog enriched
the LLM tag rung (`recommenders/llm.py`) is **not competitive** — it ties lexical
on cross-listing (0.957) and falls *below* plain TF-IDF on free text (0.404 vs
0.461); the earlier subset-run "win" (0.960/0.585) was a target/distractor
vocabulary-separation artifact, and distilling a description to ~6–12 tags loses
more signal than the LLM's normalization adds (ADR-0009, RESULTS Phase 7). SBERT
MiniLM still tops both ranking lenses. The 100% tag cache is reusable by the
deferred LLM reranker + "why this fits" UI.

## Next task

Build the Phase 7 **zero-shot LLM reranker** (`recommenders/llm.py`, plan §2.8b):
a `LLMRerankRecommender` that retrieves top-N (SBERT or TF-IDF), then reorders the
candidates with a single deterministic Ollama call over their **full text** (not
tags), reusing `OllamaClient`; cache the rerank by `sha1(model+query+candidate-ids)`,
degrade gracefully when Ollama is down (fall back to the base order), contract-test
with the `FakeClient` pattern, wire into `run_eval`, and write an ADR for the
rerank-prompt + caching design.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Next rung after the LLM reranker | "Why this fits" explanations + LLM-as-judge (finish Phase 7 b/c) / start Phase 8 Streamlit UI / try a bigger model (qwen3:32b) on the reranker | Sandeep | next session |

## Blockers / waiting-on

None. (The LLM reranker needs `ollama serve` up + `qwen3:8b` at query time; the
tag cache is already 100% warm and the rest of the suite runs fully offline.)

## First task for next session

Scaffold `LLMRerankRecommender` in `src/courserec/recommenders/llm.py`: retrieve
top-N with the MiniLM SBERT base, reorder via one deterministic Ollama call over
the candidates' full text, cache by `sha1(model+query+candidate-ids)`, fall back to
the base order when Ollama is down, add `FakeClient` contract tests, and wire it
into `scripts/run_eval.py`.
