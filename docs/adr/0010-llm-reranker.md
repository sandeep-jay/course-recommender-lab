# ADR-0010: Zero-shot LLM reranker (Phase 7 / B.8b)

**Date:** 2026-06-19
**Status:** Accepted

## Context
Track B rung 8 has three parts (plan §2.8 / §5): (a) tag extraction — shipped and
settled as *not competitive* ([ADR-0009](0009-llm-enrichment-ollama.md)); (b) a
**zero-shot LLM reranker**; (c) a "why this fits" explanation. This ADR covers
(b). ADR-0009's honest finding pointed straight here: the LLM's value lives in
operations over the **full candidate text**, not in lossy tag distillation. A
reranker is exactly that — keep a strong first-stage ranker, and spend the LLM
only on reordering its top-N candidates against the query, reading each
candidate's full text. It is the cross-encoder pattern from
[ADR-0005](0005-rerank-mmr.md) with a zero-shot LLM in place of a trained
cross-encoder. Load-bearing questions:

1. **What does the LLM see, and what does it return?** Free text it must reorder,
   not regenerate — and the reply must be parseable and bounded.
2. **Determinism + cost.** A leaderboard run must stay fast and reproducible; an
   uncached LLM call per query at eval time risks both.
3. **Graceful degradation.** The repo runs offline with no key (plan §1). What
   happens when Ollama is down — skip, or fall back?
4. **Cache identity.** The HANDOFF fixed the key as
   `sha1(model + query + candidate-ids)`; the candidate set is part of the query.

## Decision
1. **Retrieve-then-reorder, base reused wholesale.** A fitted base ranker
   (default MiniLM `SbertRecommender`, any `Recommender` injectable) retrieves the
   top `retrieve_n` (default **20**) candidates — seed already excluded by the
   base — and the LLM only **reorders** that small set. The catalog is never
   compressed (the tag rung's flaw); the LLM judges already-relevant candidates
   over their full text.
2. **Numbered listing → integer permutation, structured + validated.** Candidates
   are presented as a numbered listing (`[1] <text>` …, each text truncated to
   `candidate_chars`, default **1000**, to bound prompt size on long
   descriptions). Ollama's `format` field carries `_RERANK_SCHEMA`
   (`{"ranking": [int, …]}`), so the model returns validated integers, not prose
   to regex. Determinism pinned (`temperature=0`, `seed=RANDOM_SEED`,
   `think=False`). **Indices, not ids**, so a malformed reply is trivially
   range-checked — the model can't invent a plausible-looking wrong course id.
3. **Reconcile to a full permutation, always.** The model's indices are mapped
   back to ids, then `_reconcile` drops out-of-range/duplicate picks and appends
   any candidate the model omitted **in base order**. The output therefore ranks
   every candidate exactly once however the model (mis)behaved — a partial or
   garbled reply degrades toward the base order rather than dropping candidates.
4. **Rank-based scores.** The reranked list is scored `len - position` (strictly
   descending), honoring the interface contract independent of the base's own
   score scale.
5. **Cache by `sha1(model + normalized-query + candidate-ids)`.** Stored at
   `artifacts/llmcache/<model>/reranks.json` (`_RerankCache`). The candidate ids
   (in retrieval order) are part of the key, so any change to what the base
   retrieves invalidates the entry rather than reusing a stale order. One
   deterministic call per (query, candidate-set), ever.
6. **Fall back, don't fail — skip only when cold + offline.** When Ollama is
   unreachable the rung returns the **base order** for any uncached query (worst
   case "no better than the base," never a crash). `fit` raises `LLMUnavailable`
   (harness skips + flags) **only** in the cold case — Ollama down *and* the
   rerank cache empty — where every query would reproduce the base and the
   leaderboard row would be a useless duplicate. With a warm cache it runs fully
   offline.
7. **Zero new dependencies.** Reuses the stdlib-`urllib` `OllamaClient` from
   ADR-0009, adding only `rank_candidates`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Model returns ranked **ids** | Readable cache/reply | Model can emit plausible-but-wrong ids; hard to validate | Integers are range-checkable; reconcile is trivial |
| Free-text "list them best first" | No schema needed | Brittle parsing, nondeterministic format | `format` schema gives a validated permutation |
| Per-candidate 0–1 relevance score | Fine-grained; sums to a blend | N separate calls or a wide schema; scale drift across queries | One call returning an order is cheaper and sufficient for ranking |
| Skip the rung when Ollama is down (like the tag rung) | Consistent with ADR-0009 | A reranker *can* serve the base order offline; skipping discards a warm cache | Fall back to base; skip only cold + offline |
| `fit` reranks the whole eval set eagerly | Warms cache in one place | Couples fit to the query set; slow, query-dependent | Reranking is query-time; cache fills lazily |
| Drop omitted candidates from the model's reply | Honors the model literally | Returns < k, violates "rank every candidate" intent | Reconcile appends omissions in base order |

## Consequences
**Positive:** The LLM is spent where ADR-0009 said its value is — over full
candidate text, not distilled tags. No catalog compression, so no
information-loss failure mode. Deterministic and cached, so a leaderboard run
stays fast and reproducible; offline runs reuse a warm cache and otherwise
degrade to the base order instead of failing. Reuses the existing client + cache
machinery (zero new deps) and the base's own artifact, and `_reconcile` makes the
rung robust to any malformed model reply.

**Negative / honest caveats:** (1) Quality is **unmeasured at write time** — this
ADR ships the mechanism; the leaderboard delta vs the SBERT base needs a warm-cache
`run_eval` with `ollama serve` up (next session). The reranker can only reorder
what the base retrieves, so its ceiling is the base's recall@`retrieve_n`. (2)
`candidate_chars=1000` truncates the longest descriptions (catalog max ~181 words
≈ within budget for most, but not all) — a deliberate prompt-size bound, not a
free choice. (3) `retrieve_n=20`, `candidate_chars`, and the prompt wording are
chosen defaults, not tuned — obvious future sweeps. (4) Latency: one LLM call per
distinct (query, candidate-set) on a cold cache; the cross-listing + judged-query
eval sets are bounded, so the one-time cost is minutes, not hours.

**Neutral:** Default base is MiniLM SBERT (the top ranking-lens rung); a TF-IDF
base is a one-arg swap to isolate the rerank's contribution from the retriever's.

## Implementation notes
`src/courserec/recommenders/llm.py`: `OllamaClient.rank_candidates` (numbered
listing → integer permutation), `_RerankCache`
(`sha1(model+query+candidate-ids)` → `reranks.json`), and `LLMRerankRecommender`
(base retrieval, `_reconcile` to a full permutation, rank-based scores,
base-order fallback, cold+offline skip). Wired into `scripts/run_eval.py`
alongside `LLMTagRecommender` under the existing `LLMUnavailable` graceful-skip
`except`. Tests: `tests/test_llm.py` (`FakeBase` + `FakeRerankClient`, no daemon)
— reorder, seed-exclusion, sort/cap, cache reuse, cold-offline skip, warm-offline
cache, uncached-offline base fallback, malformed-output reconcile, by-text,
empty-query, unknown-seed, bad-config. Builds on [ADR-0005](0005-rerank-mmr.md)
(retrieve→rerank) and [ADR-0009](0009-llm-enrichment-ollama.md) (Ollama client +
cache); the "why this fits" explanation (B.8c) remains deferred.
