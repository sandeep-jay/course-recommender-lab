# HANDOFF — course-rec-bert

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Claude Code setup is now project-correct: lakehouse leftovers purged, CLAUDE.md +
rules + commands retargeted to the content-based course recommender, and git is
initialized (branch `main`, nothing committed by the protocol yet — the initial
commit happens below). No application code, tests, scripts, or data pipeline exist
yet; the contract is [docs/roadmap/recommender_plan.md](docs/roadmap/recommender_plan.md).

## Next task

Resolve the data-path discrepancy, then build Phase 0: move
`data/courses-report.2026-06-02.csv` → `data/raw/courses-report_2026-06-02.csv`
(or pick one canonical path and update CLAUDE.md + the plan), then create
`pyproject.toml`, `src/courserec/{interfaces.py,data.py}`, `scripts/prepare_data.py`,
and `tests/` per recommender_plan.md §5 Phase 0.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Canonical data path | `data/raw/courses-report_2026-06-02.csv` (per plan) vs current `data/courses-report.2026-06-02.csv` | Sandeep | Phase 0 |
| Project name | `course-rec-bert` (dir) vs `course-rec-lab` (plan) | Sandeep | Phase 0 |

## Blockers / waiting-on

None.

## First task for next session

Reconcile the data path (move the CSV into `data/raw/` with the plan's filename) and scaffold Phase 0 per recommender_plan.md §5.
