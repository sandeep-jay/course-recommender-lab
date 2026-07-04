# HANDOFF — course-recommender-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Roadmap phases **0–8 complete and green** — `pytest` = **211 passed**, `ruff`/`black`
clean — with the Streamlit UI shipping as a warm offline Docker image and ten teaching
notebooks. The **docs site is live** at https://sandeep-jay.github.io/course-recommender-lab/,
and **two CI workflows now gate every push**: `test.yml` (lint + pytest + notebook
render-freshness) and `docs.yml` (strict build + Pages deploy), both on the node24 action
runtime. Only optional polish remains — no functional work is pending.

## Next task

Add tests + docs-deploy status badges to the top of `README.md`, using the shields
endpoints `https://github.com/sandeep-jay/course-recommender-lab/actions/workflows/test.yml/badge.svg`
and `.../docs.yml/badge.svg`, so the passing CI is visible to reviewers landing on the repo.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Build Stage 3 concept primer | a gentle `docs/concepts.md` RecSys intro vs skip (Start Here already folds in the core idea) | Sandeep | optional |
| Docs dep drift | reconcile `requirements-docs.txt` (ranges → material 9.7.6 / pymdownx 10.21.3) and the `[docs]` extra (exact pins → 9.5.49 / 10.14) to one style; both build clean today | Sandeep | when convenient |
| Where to host the Docker image | Cloud Run / HF Spaces (Docker) / leave as a local `docker run` portfolio artifact | Sandeep | when convenient |
| pyproject **import** package name | keep `courserec` (hyphens illegal in module names) vs rename to `course_recommender_lab` (~228 sites) | Sandeep | not planned |

## Blockers / waiting-on

None.

## First task for next session

Add the tests + docs-deploy CI status badges to the top of `README.md`.
