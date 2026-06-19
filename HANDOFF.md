# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Phases 0–6 plus metadata fusion (Phase 5 / B.5) and the judged free-text lens are
green; `python scripts/run_eval.py` regenerates the three leaderboards and
`python scripts/run_clustering.py` regenerates the Phase 6 diagnostic, with
`pytest` = **135 passed** and `ruff`/`black` clean. Metadata fusion
(`src/courserec/recommenders/metadata.py`, a `Recommender`) fuses a TF-IDF text
block with a weighted one-hot subject+department+level+units block (score
`λ²·cos_text + (1−λ)²·cos_meta`); it **loses** on the cross-listing lens
monotonically in λ (0.948→0.936→0.909 vs 0.955 TF-IDF) because 99.7% of
cross-listing edges span different subjects, and ties the baseline exactly on the
free-text lens (zero metadata in a query) — an honest negative documented in
ADR-0008. Track A/B techniques remaining are Phase 7 LLM enrichment and Phase 8
the Streamlit UI.

## Next task

**Phase 7 — LLM enrichment** (recommender_plan.md §2.8, §5):
`src/courserec/recommenders/llm.py` — (a) LLM extracts structured tags
(topics/skills/level/prereqs-mentioned) per course → richer features, (b) a
zero-shot LLM reranker over candidate sets, (c) a one-line "why this fits"
explanation for the UI. **Must degrade gracefully with no API key** (skip + flag,
never hard-fail the suite — plan §1), like the API embedding rung.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Phase 7 LLM provider | Anthropic (Claude) vs. OpenAI — pick the SDK + key env for tag extraction / zero-shot rerank, keeping the no-key graceful-skip path | Sandeep | next session |

## Blockers / waiting-on

None.

## First task for next session

Scaffold `src/courserec/recommenders/llm.py` per the `/new-recommender` contract:
start with the **tag-extraction → richer-feature** path (cache extracted tags to
`artifacts/<name>/`, key on `sha1(model + normalized_text)` like the embedding
cache), subclass `Recommender`, raise the graceful-skip exception when no key is
set, add a contract test, and wire it into `scripts/run_eval.py`. Write an ADR for
the provider/caching choice.
