# Rules: src/courserec/recommenders/

These rules apply when editing any technique in `src/courserec/recommenders/`.
The contract is `docs/roadmap/recommender_plan.md` §1–2.

## Interface
- Every technique subclasses `Recommender` (`src/courserec/interfaces.py`) and
  sets a unique `name` (used in the leaderboard) and a `config` dict (hyperparams,
  logged with results).
- Implement `fit(courses)`, `recommend_similar(course_id, k)`, and
  `recommend_by_text(query, k)`. A text-incapable, item-to-item-only technique
  may raise `NotImplementedError` from `recommend_by_text` — never return garbage.
- Both recommend methods return `list[Rec]` (`course_id`, `score`), sorted by
  descending score, length ≤ k.

## Hard rules (violations are bugs)
- **Exclude the seed:** `recommend_similar` MUST NOT include the seed `course_id`
  in its results.
- **No leakage:** never read `Cross-Listed Course(s)` as an input feature — it is
  the evaluation ground truth. The graph technique is the sole exception and must
  evaluate only on a held-out edge split.
- **Sparse text:** some courses have a 1-word or missing description — fall back to
  the title; never crash on empty text.

## Artifacts & reproducibility
- Persist fitted artifacts (vectors, indexes, embedding caches) to
  `artifacts/<name>/` and load them if present — never recompute embeddings on
  every run. `artifacts/` is gitignored.
- Embedding cache key = `sha1(model_name + normalized_text)`.
- Use the global `RANDOM_SEED = 42` for any stochastic step.
- API-backed techniques (API embeddings, LLM) must degrade gracefully when no key
  is set — skip and note it, never hard-fail the suite.

## Tests
- Every technique gets a test in `tests/` asserting the contract: seed excluded,
  returns `list[Rec]` of length ≤ k, scores sorted descending, handles a
  sparse-text course without crashing.
