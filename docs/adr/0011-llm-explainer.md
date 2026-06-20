# ADR-0011: "Why this fits" explainer (Phase 7 / B.8c)

**Date:** 2026-06-20
**Status:** Accepted

## Context
Track B rung 8 has three parts (plan §2.8 / §5): (a) tag extraction — shipped and
settled as *not competitive* ([ADR-0009](0009-llm-enrichment-ollama.md)); (b) a
zero-shot LLM reranker — shipped and settled as a second *negative result*
([ADR-0010](0010-llm-reranker.md)); (c) a one-line **"why this fits"
explanation**. This ADR covers (c), the last B.8 piece. The two negative results
are not a dead end — they are the evidence pointing here. Both (a) and (b) tried
to make the LLM **rank**, and a 0.38 M-param SBERT bi-encoder already ranks at the
recall ceiling, so the LLM had no headroom to add. Explanation is a different job:
the ranking already exists (from the top rung, SBERT), and the LLM is spent only on
**justifying** one pair — a generative task with no ground-truth ordering to beat.
Load-bearing questions:

1. **Is this a recommender?** It produces no ranking. Where does it sit relative to
   the `Recommender` interface, the eval harness, and the leaderboard?
2. **What does the LLM see and return?** A (query, candidate) pair → one bounded,
   parseable sentence — not prose to regex.
3. **Determinism + cost + caching.** Same reproducibility and offline-runs-clean
   constraints as ADR-0009/0010.
4. **Graceful degradation.** The "why" line is explicitly *optional* in the UI
   (plan §4: "if LLM enrichment ran"). What happens when Ollama is down?

## Decision
1. **A presentation helper, not a `Recommender`.** `RecommendationExplainer` is
   deliberately **not** a `Recommender` subclass: it returns no `list[Rec]`, has no
   `recommend_*` methods, and is **never scored by `eval.py` or added to the
   leaderboard**. An explanation has no ground-truth ranking to measure — scoring
   it would be category error. It layers on top of whatever rung the UI is showing.
   `fit(courses)` captures `text`/`title` maps and opens the cache (returning
   `self` for one-line construction); no LLM call happens at fit time.
2. **One bounded sentence, structured + validated.** `OllamaClient.explain` sends
   the query and the candidate's title + text (each truncated to `candidate_chars`,
   default **1000**, to bound prompt size) with Ollama's `format` field carrying
   `_EXPLAIN_SCHEMA` (`{"reason": str}`), so the reply is a validated string, not
   prose to parse. Determinism pinned (`temperature=0`, `seed=RANDOM_SEED`,
   `think=False`); the result is whitespace-normalized.
3. **Cache by `sha1(model + normalized-query + candidate-id)`.** Stored at
   `artifacts/llmcache/<model>/explanations.json` (`_ExplanationCache`). The
   candidate **id** (not its text) is part of the key, matching the rerank cache —
   the catalog is static, so the id pins the text. One deterministic call per
   (query, candidate), ever.
4. **Degrade to `None`, never raise on unavailability.** Because the "why" line is
   optional, *every* unavailability path returns `None` (the UI simply omits the
   line): an empty query, a blank model reason, or Ollama unreachable with no
   cached entry. `fit` **never** skips/raises `LLMUnavailable` (unlike the two
   ranking rungs, which would otherwise be useless leaderboard duplicates — an
   explainer that returns no line is simply quiet, not broken). The one hard error
   is a programming bug: an unknown `candidate_id` raises `KeyError`.
5. **Item-to-item convenience.** `explain_seed(seed_id, candidate_id)` resolves the
   seed's own text as the query, so the UI's "similar courses" mode needs no
   text plumbing.
6. **Zero new dependencies.** Reuses the stdlib-`urllib` `OllamaClient` from
   ADR-0009, adding only `explain`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Make it a `Recommender` (rank by "explainability") | Uniform with every rung; auto-leaderboarded | No ground-truth ranking; would invent a metric to optimize a non-ranking task | It produces no ordering — scoring it is a category error |
| Free-text reason (no schema) | No schema to define | Brittle parsing, preamble/"thinking" leakage | `format` schema gives one validated string |
| Skip the rung when Ollama is down (like ADR-0009/0010) | Consistent with the ranking rungs | An *optional* UI line has no "useless duplicate" failure mode — skipping a whole fit is heavier than omitting a line | Degrade per-call to `None` instead |
| Key the cache on candidate **text** | Survives a catalog edit | Heavier key; the catalog is static | Id pins the text here, matching `_RerankCache` |
| Batch-generate every (seed, candidate) pair eagerly | Warms the cache up front | Combinatorial (~11k × k); couples to a query set | Generate lazily at query time, cache per pair |
| One call explaining *all* top-k together | Fewer calls | Couples reasons; a changed k reshuffles the prompt and busts the cache for every pair | One call per pair caches independently |

## Consequences
**Positive:** The LLM is finally spent on a task it is actually good at and that
the two negative results pointed to — generating a human justification, not
competing with SBERT on ranking. Clean separation of concerns: rankers rank and are
scored; the explainer explains and is not. Deterministic and cached, so it adds
nothing to leaderboard runtime (it is not in the leaderboard at all) and the repo
still runs end-to-end offline — a cold, offline explain is a quiet no-op, not a
crash. Reuses the existing client + cache machinery (zero new deps).

**Validated live (2026-06-20, qwen3:8b).** `scripts/explain_recs.py` over SBERT
recommendations produces concrete, on-topic one-liners in both modes, e.g.
`COMPSCI 189 → STAT C241A`: *"Both courses cover statistical learning theory,
classification, regression, clustering, dimensionality reduction, and ensemble
methods."*; query *"ethics of artificial intelligence" → PHILOS 14*: *"…explores
ethical issues in AI, such as algorithmic bias and moral responsibility…"*. The
cache persists and a repeat (query, candidate) serves instantly without a call.

**Honest caveats:** (1) **No automatic quality metric** — by construction, there is
no ground-truth set for free-text justifications, so quality is assessed by
inspection only (an LLM-as-judge over a sample is a possible future check, itself
needing validation per plan §3). (2) The output is **qwen3:8b-specific** and
occasionally restates the query rather than naming the *shared* concept; prompt
wording and `candidate_chars` are chosen defaults, not tuned. (3) Explanations are
generated against whatever rung the UI shows — they describe a recommendation, they
do not certify it is correct.

**Neutral:** No leaderboard row, no `run_eval.py` wiring — the explainer is invoked
by the future Phase 8 UI and by `scripts/explain_recs.py`, not by the eval harness.

## Implementation notes
`src/courserec/recommenders/llm.py`: `OllamaClient.explain` (pair → one validated
sentence), `_ExplanationCache` (`sha1(model+query+candidate-id)` →
`explanations.json`), and `RecommendationExplainer` (`fit` captures text/title maps
+ opens the cache; `explain`/`explain_seed` with cache→live→`None` degradation,
`KeyError` only on an unknown id). Driver: `scripts/explain_recs.py`
(`--seed`/`--query`, SBERT base, live Ollama or clear exit). Tests:
`tests/test_llm.py` (`FakeExplainClient`, no daemon) — reason propagation,
seed-text resolution, cache reuse, warm-offline cache hit, cold-offline `None`,
empty-query `None`, blank-reason `None`, unknown-candidate `KeyError`,
before-`fit` `RuntimeError`, bad-config. Builds on
[ADR-0009](0009-llm-enrichment-ollama.md) (Ollama client + cache) and closes
Track B.8 after [ADR-0010](0010-llm-reranker.md). **Completes rung 8.**
