# ADR-0009: LLM enrichment via local Ollama (Phase 7 / B.8, tag-extraction rung)

**Date:** 2026-06-19
**Status:** Accepted

## Context
Track B rung 8 (plan §2.8 / §5) is LLM enrichment in three parts: (a) extract
structured tags per course → richer features, (b) a zero-shot LLM reranker, (c) a
"why this fits" explanation for the UI. This ADR covers the **provider + caching +
first rung (tag extraction)**; (b) and (c) are deferred. Decisions that were
load-bearing:

1. **Provider.** The open decision (HANDOFF) was Anthropic vs OpenAI. Both are
   cloud APIs needing a key — in tension with the repo's hard constraint that it
   run end-to-end with no API key (CLAUDE.md rule 5, plan §1).
2. **When the LLM runs.** ~11k courses × seconds/call is hours. If `fit` generated
   on demand, every eval run risks an hours-long, nondeterministic stall.
3. **How tags become a ranking.** Extracted tags are free-form strings; they need
   to become a comparable feature space without simply re-deriving raw text.
4. **Graceful degradation.** What does "no key" mean when there is no key — when
   the dependency is a *local daemon* that may or may not be running?
5. **Dependency budget.** The plan keeps heavy/optional deps behind extras.

## Decision
1. **Local Ollama, not a cloud API.** The model runs on-device via Ollama's HTTP
   API (`localhost:11434`) — no key, no per-call cost, no network beyond
   localhost. This *strengthens* the local-only guarantee instead of bending it:
   the graceful-skip trigger becomes "Ollama unreachable / nothing enriched"
   rather than "no key." Default model **qwen3:8b** (verified: clean structured
   JSON, ~5 s/course); larger models (`qwen3:32b`, `gemma4:31b`) are a one-flag
   swap. Chosen because Sandeep runs Ollama locally and asked to use it.
2. **Structured output via JSON schema.** Ollama's `format` field carries
   `_TAG_SCHEMA`, so the model returns a validated `CourseTags` object
   (topics/skills/level/prereqs_mentioned), not free text to regex. Determinism is
   pinned (`temperature=0`, `seed=RANDOM_SEED`, `think=False`).
3. **Enrichment is a separate, explicit, resumable pass.** `fit` **never** calls
   the LLM — it only *reads* the tag cache. The slow generation lives in
   `enrich_courses` / `scripts/enrich_catalog.py`, which skips already-cached ids
   (interrupt-and-resume) and by default enriches just the **eval-relevant
   subset** (cross-listing seeds + twins + judged-query gold ≈ 1.3k courses,
   minutes), `--all` for the full catalog. So evaluation stays fast and
   deterministic and the expensive pass is a deliberate one-time cost.
4. **TF-IDF cosine over tag *profiles*, with raw-text fallback.** Each course's
   tags flatten to a profile string (topics+skills+prereqs; `level` excluded — the
   metadata rung already owns it); profiles are TF-IDF-vectorized and compared by
   cosine, exactly like the lexical rung but over the LLM's distilled vocabulary.
   A course with no cached tags falls back to its raw text, so the rung never
   crashes on a cold course — it just loses distillation for it.
5. **Skip only when it would be a useless duplicate.** If no course is enriched
   *and* Ollama is unreachable, `fit` raises `LLMUnavailable` (harness skips +
   flags) rather than silently shipping a raw-text TF-IDF clone. With a warm cache
   it runs offline; with Ollama up but a cold cache it falls back to text and logs
   a warning.
6. **Zero new dependencies.** The client is stdlib `urllib` — no `openai`/
   `ollama`/`requests`. Cache key is `sha1(model + normalized_text)` (the rule's
   key), stored at `artifacts/llmcache/<model>/tags.json`.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Anthropic / OpenAI cloud API | Strongest models; trivially parallel | Needs a key + network + spend; the repo must run keyless | Ollama is keyless, local, free, and Sandeep asked for it |
| `fit` generates tags on demand | One step; always fresh | Hours-long, nondeterministic eval runs; re-generates every run | Cache-read fit + explicit enrich pass keeps eval fast/deterministic |
| Multi-hot over exact tag strings | Interpretable indicators | Free-form tags rarely match verbatim across courses | TF-IDF over profile tokens matches at the token level, more forgiving |
| Embed tag profiles with SBERT | Best cross-course matching | Pulls the `semantic` extra into the LLM rung; couples two phases | TF-IDF keeps the rung self-contained; a clean delta vs the `tfidf` baseline |
| `ollama` / `openai` python SDK | Convenience helpers | A dependency for a couple of HTTP POSTs | stdlib `urllib` is enough; zero new deps |
| Include `level` in the profile | One more signal | Coarse 4-way band, low discriminative power, already a metadata facet | Keep the profile to conceptual content; avoid double-counting |

## Consequences
**Positive:** A keyless, local, zero-cost, zero-new-dependency LLM rung that keeps
the repo's offline guarantee intact and even strengthens it. Enrichment is
cached, resumable, and decoupled from eval, so a leaderboard run is still fast and
deterministic. The same machinery unblocks the deferred pieces — the `nomic-embed-text`
model already pulled locally can later back an Ollama-embedding rung, and the
zero-shot reranker + "why this fits" reuse this client and cache.

**Negative / honest finding — full enrichment overturns the provisional win; the
rung is not competitive.** The subset-enriched run (12.5%) had `llm_tags(qwen3_8b)`
beating *every* lexical baseline on both lenses (cross-list 0.960, free-text
0.585). We then ran `scripts/enrich_catalog.py --all` (full catalog, ~5 h, cached)
to enrich the distractors too — exactly to test that number — and the verdict
flips:

| Lens | partial (12.5%) | **full (100%)** | tfidf baseline |
|---|---|---|---|
| cross-listing NDCG@10 | 0.960 | **0.957** (tied with lexical, mid-pack) | 0.955 |
| free-text NDCG@10 | 0.585 | **0.404** (*below* tfidf) | 0.461 |

So the apparent lift was almost entirely the **target/distractor
vocabulary-separation artifact**: with only the eval targets in tag space they were
trivially separable from raw-text distractors; once everything is tag space, the
cross-listing edge vanishes (twins share near-identical tags just as they share
near-identical text) and the free-text number drops *below* plain TF-IDF. The
mechanism is information loss: compressing a 50–200-term description into ~6–12
abstract tags discards discriminative detail that TF-IDF exploits, and the tag
vocabulary saturates at 11k courses — the loss outweighs the synonym-normalization
the LLM adds. Extraction quality was never the issue (spot checks are clean); the
*architecture* — lossy distillation then lexical matching — is. The boards now
carry an "LLM enrichment (full)" note (the number is comparable, no caveat). The
takeaway: the LLM's value lives in operations over the *full* candidate text (the
deferred zero-shot reranker and "why this fits"), not in tag distillation — and
running `--all` before believing the win was the whole point.

**Neutral:** qwen3:8b and the profile composition (topics+skills+prereqs, TF-IDF)
are chosen defaults, not tuned; model size and profile design are obvious future
sweeps. Tag quality is not separately validated against human labels here — the
LLM-as-judge piece (plan §3) is the future mechanism for that.

## Implementation notes
`src/courserec/recommenders/llm.py`: `OllamaClient` (stdlib urllib, `available` +
`extract_tags`), `_TagCache` (`sha1(model+text)` → `tags.json`), `CourseTags`,
`enrich_courses`, and `LLMTagRecommender` (cache-read `fit`, TF-IDF profiles,
raw-text fallback). Driver: `scripts/enrich_catalog.py` (eval-relevant subset by
default, `--all` / `--model`). Wired into `scripts/run_eval.py` with
`LLMUnavailable` joining the graceful-skip `except`. Config: `OLLAMA_HOST`,
`DEFAULT_LLM_MODEL` in `src/courserec/config.py`. Tests: `tests/test_llm.py`
(FakeClient, no daemon). Builds on the lexical rung
([ADR-0002](0002-eval-harness-design.md)); complements the metadata rung's clean
ablation discipline ([ADR-0008](0008-metadata-fusion.md)).
