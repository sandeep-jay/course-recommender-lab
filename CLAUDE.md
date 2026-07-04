# course-recommender-lab — Claude Code Configuration

## Project
A sandbox for learning recommender systems by implementing, documenting, and
comparing **content-based** techniques on the UC Berkeley course catalog
(~11,091 courses). No user-interaction data exists, so collaborative filtering
is **out of scope**. Every technique conforms to one interface, is scored by one
shared evaluation harness, and ranked on a leaderboard.

The full contract is [docs/roadmap/recommender_plan.md](docs/roadmap/recommender_plan.md).
Read it before implementing — it defines the phases, interface, and eval methodology.

## Load-bearing rules (easy to get wrong, expensive to catch)
1. **Null token:** the catalog uses the string `"-"` as its null value. Replace
   with real NA on load — never treat `"-"` as data.
2. **Leakage:** `Cross-Listed Course(s)` is the **evaluation ground truth**. No
   technique may read it as an input feature (the graph model is the one
   exception and must use a held-out edge split).
3. **Interface:** every technique subclasses `Recommender` (`src/courserec/interfaces.py`).
   `recommend_similar` MUST exclude the seed course from its own results.
4. **Artifacts:** fitted models / embedding caches persist to `artifacts/<name>/`
   and load if present — never recompute embeddings every run. `artifacts/` is gitignored.
5. **Reproducibility:** global `RANDOM_SEED = 42`. The repo must run end-to-end
   through Phase 6 with **no API key**; LLM/API phases degrade gracefully if absent.

## Data
- Catalog CSV currently at `data/courses-report.2026-06-02.csv`.
  (The plan references `data/raw/courses-report_2026-06-02.csv` — reconcile the
  path before writing the loader; pick one canonical location.)
- Parse with pandas, never line-based — descriptions contain RFC-4180 quoted newlines.
- `course_id` is synthesized as `f"{Subject} {Course Number}"` (e.g. `AEROENG 1`).
- Some courses have 1-word or missing descriptions — fall back to title, never crash.

## How to run
- Tests: `pytest` (tests live in `tests/`)
- Lint / format: `ruff check .` / `black .`
- Eval + leaderboard: `python scripts/run_eval.py` (regenerable in one command)
- Full operational runbook (every script/flag, models, install tiers,
  troubleshooting): [docs/RUNBOOK.md](docs/RUNBOOK.md)

## Code quality defaults (per global config)
- Type hints on signatures, Google-style docstrings on public functions.
- `logging`, not `print`, in library code. Functions under ~40 lines.
- Pinned deps in `pyproject.toml`. No hardcoded paths, keys, or magic numbers.

## Session protocol (per global config)
- Start: read [HANDOFF.md](HANDOFF.md) (`/session-start`).
- End: `/session-end` — update HANDOFF + CHANGELOG, run tests, write any ADRs, commit.
- Architectural decisions get an ADR in `docs/adr/` (`/new-adr`).

## Skills / commands (.claude/)
  /session-start     — Read HANDOFF, summarize state, begin next task
  /session-end       — Full end-of-session protocol
  /new-adr           — Write an ADR + update the index
  /new-recommender   — Scaffold a technique + contract test against the interface
