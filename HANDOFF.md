# HANDOFF — course-rec-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Roadmap phases **0–8 complete and green**: `pytest` = **211 passed**, `ruff`/`black`
clean; the Phase 8 UI also ships as a warm offline Docker image and has ten
step-by-step teaching notebooks under `notebooks/` (all execute under
`pytest --nbmake`). The only open thread is purely optional — where (if anywhere) to
host the Docker image; no roadmap work pends.

## Next task

Optional. To publish the UI: push the `course-rec-ui` image to a registry and deploy
to the chosen host (it's already `$PORT`-aware/headless — no code change). Otherwise
wind down to docs polish. No required task remains.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Where to host the Docker image | Cloud Run / HF Spaces (Docker) / leave as a local `docker run` portfolio artifact | Sandeep | when convenient |

## Blockers / waiting-on

None.

## First task for next session

Decide the host row above; if undecided, polish `README.md`/`docs/RESULTS.md` for
portfolio reading and link the `notebooks/` walkthroughs from the README lead.
