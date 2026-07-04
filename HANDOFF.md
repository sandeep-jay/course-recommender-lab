# HANDOFF — course-recommender-lab

> State only. For what happened this (or any prior) session see [CHANGELOG.md](CHANGELOG.md).
> Narrative belongs there, not here.

## Current state

Roadmap phases **0–8 complete and green**: `pytest` = **211 passed**, `ruff`/`black`
clean; the Phase 8 UI also ships as a warm offline Docker image and has ten
step-by-step teaching notebooks under `notebooks/` (all execute under `pytest --nbmake`).
The repo is now on a **private GitHub remote** (`sandeep-jay/course-recommender-lab`)
and has a **MkDocs Material docs site** (ADR-0015) that builds `--strict` clean and
renders the notebooks as pre-executed pages. The project was renamed to
`course-recommender-lab` (import package stays `courserec`).

The local working dir has now **been renamed** to `course-recommender-lab`; the
`.venv` that broke as a result (stale `course-rec-bert` paths in the editable install
and every `.venv/bin/*` shebang) was repaired in place — no dependency reinstall —
and the rename regressions to the `ruff check .`-clean invariant were fixed. See the
CHANGELOG "Fixed" entries.

## Next task

**One-time manual step to publish the docs site:** GitHub → repo **Settings → Pages →
Source = "GitHub Actions"**. The `docs.yml` workflow then deploys on push to
`https://sandeep-jay.github.io/course-recommender-lab/`. After that, optional: decide
Docker-image hosting (below), and optionally rename the local working directory to match.

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| Enable GitHub Pages (Actions source) | one click in Settings → Pages | Sandeep | to publish the site |
| ~~Rename local working dir `course-rec-bert` → `course-recommender-lab`~~ | **Done** — dir renamed, `.venv` repaired. Only loose end: the `.claude` memory path still keys off the old dir name if you rely on it. | Sandeep | resolved |
| pyproject **import** package stays `courserec` | keep (hyphens illegal in module names) vs rename to `course_recommender_lab` (228 sites) | Sandeep | not planned |
| Where to host the Docker image | Cloud Run / HF Spaces (Docker) / leave as a local `docker run` portfolio artifact | Sandeep | when convenient |

## Blockers / waiting-on

None.

## First task for next session

Flip the Pages source to "GitHub Actions" and confirm the site deploys; then decide the
Docker-host row. If regenerating the notebook pages, run `make docs-notebooks` locally
(needs the catalog + optional Ollama) before pushing.
