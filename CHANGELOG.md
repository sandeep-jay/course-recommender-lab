# Changelog

All notable changes to course recommender implementation
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- Project-specific Claude Code rules: `.claude/rules/recommenders.md` (interface
  contract, seed exclusion, no-leakage, artifacts) and `.claude/rules/eval.md`
  (leakage discipline, three lenses, bootstrap CIs, regenerable leaderboard).
- `/new-recommender` command — scaffolds a technique + contract test + scoring,
  replacing the lakehouse `/new-transform`.
- `.gitignore` (artifacts, `*.npz`, `data/processed/`, `.env*`, `settings.local.json`,
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

### Removed
- Lakehouse leftovers copied from scribe-iq-lakehouse: `.claude/rules/{fabric-transforms,
  notebooks,transforms}.md`, `.claude/skills/{delta-patterns,healthcare-data}.md`,
  `.claude/commands/new-transform.md`.

