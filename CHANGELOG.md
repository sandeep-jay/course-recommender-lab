# Changelog

All notable changes to course recommender implementation
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- **Phase 0 — scaffold & data.** `pyproject.toml` (package `course-rec-lab`,
  pinned pandas/numpy/pyarrow + dev tools, ruff/black/pytest config); `src/courserec/`
  with `config.py` (paths, `RANDOM_SEED=42`, `NULL_TOKEN`), `interfaces.py`
  (`Rec`, `Recommender` ABC with the seed-exclusion / no-leakage contract), and
  `data.py` (load → null-token→NA → `course_id` → level parsing → sparse-text
  fallback → duplicate-id collapse → parquet).
- `scripts/prepare_data.py` — one-command raw CSV → `data/processed/courses.parquet`.
- `tests/test_data.py` — 18 tests: row count, no `"-"` remains, id synthesis +
  uniqueness, level bands, sparse-text fallback, cross-listed sparsity.
- `docs/adr/0001-duplicate-course-ids.md` + ADR index — decision to collapse the
  16 colliding `course_id`s (34 rows) to one representative each, coalescing
  `cross_listed`. Catalog: 11,091 raw rows → **11,073 unique courses**.
- `README.md` (setup, run, Phase-0 status).
- Project-specific Claude Code rules: `.claude/rules/recommenders.md` (interface
  contract, seed exclusion, no-leakage, artifacts) and `.claude/rules/eval.md`
  (leakage discipline, three lenses, bootstrap CIs, regenerable leaderboard).
- `/new-recommender` command — scaffolds a technique + contract test + scoring,
  replacing the lakehouse `/new-transform`.
- `.gitignore` (artifacts, `*.npz`, `data/`, `.env*`, `.claude/`,
  Python/tooling/OS noise). Initialized git repo on branch `main`.

### Changed
- Rewrote `CLAUDE.md` for the course recommender (5 load-bearing rules, run
  commands, data notes) — was an empty scribe-iq-lakehouse template.
- Trimmed `.claude/settings.json` to pytest/ruff/black/git/fs allows; dropped
  AWS-S3 and detect-secrets entries.
- Reset `.claude/settings.local.json` to an empty allow list (kept additionalDirectories).
- Slimmed `.claude/hooks/scan-secrets.sh` to AWS + generic patterns; added
  OpenAI/Anthropic key patterns; dropped Azure/OneLake/MLflow/DagsHub/Fabric.
- Retargeted `/session-start` (Session-1 fallback → recommender_plan.md) and
  `/session-end` (doc-sync → leaderboard/TRADEOFFS/RESULTS).
- Fixed `HANDOFF.md` title.
- Untracked `.claude/` and `data/` from git and broadened `.gitignore` to ignore
  both wholesale (was only `settings.local.json` and `data/processed/`).
- Moved the catalog CSV to its canonical path `data/raw/courses-report_2026-06-02.csv`
  (resolving the plan-vs-disk discrepancy); chose `course-rec-lab` as the package name.

### Removed
- Lakehouse leftovers copied from scribe-iq-lakehouse: `.claude/rules/{fabric-transforms,
  notebooks,transforms}.md`, `.claude/skills/{delta-patterns,healthcare-data}.md`,
  `.claude/commands/new-transform.md`.

