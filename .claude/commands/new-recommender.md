Implement a new recommender technique for: $ARGUMENTS

Follow this sequence. The contract is `docs/roadmap/recommender_plan.md` §1–2 and
the rules in `.claude/rules/recommenders.md`.

1. **Read the interface** — `src/courserec/interfaces.py` (`Recommender` ABC, `Rec`).
   Confirm the technique's family in the plan (Track A similarity ladder or
   Track B complementary methods) and what it should be good/bad at.

2. **Create the technique** in `src/courserec/recommenders/{name}.py`:
   - Subclass `Recommender`; set a unique `name` and a `config` dict (all hyperparams).
   - Implement `fit`, `recommend_similar`, `recommend_by_text` (raise
     `NotImplementedError` from the text method only if the technique is genuinely
     item-to-item only).
   - **Exclude the seed** from `recommend_similar`. Return `list[Rec]`, sorted by
     descending score, length ≤ k.
   - **No leakage** — never read `Cross-Listed Course(s)` as a feature (graph is the
     documented exception with a held-out edge split).
   - Persist fitted artifacts to `artifacts/{name}/` and load if present — never
     recompute embeddings every run. Use `RANDOM_SEED = 42` for stochastic steps.
   - Handle sparse-text courses (fall back to title; never crash).
   - Type hints + a Google-style docstring covering: the idea, a one-line math
     sketch, complexity, and when it wins / loses.

3. **Register it** so the eval harness and UI can discover it (the technique
   registry / list used by `scripts/run_eval.py`).

4. **Create the test** in `tests/test_{name}.py` asserting the contract:
   seed excluded, returns `list[Rec]` (len ≤ k, scores descending), handles a
   sparse-text course, artifact round-trips (fit → persist → load). Run
   `pytest tests/test_{name}.py` before continuing.

5. **Score it** — run `python scripts/run_eval.py` so the technique lands on the
   leaderboard with its config and metrics.

6. **Update docs** — add a row to `docs/TRADEOFFS.md` (quality, speed,
   interpretability, cost, cold-start, complexity, when to prefer it) and note
   results in `docs/RESULTS.md`. If a non-obvious decision was made, write an ADR
   with /new-adr.

7. **Update CHANGELOG.md** with the new technique.

Do not proceed to scoring until the contract test passes.
