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

**The docs site is live**: GitHub Pages (Actions source) is enabled and `docs.yml`
deploys on push to `https://sandeep-jay.github.io/course-recommender-lab/`. A
**learner-navigation pass** landed on top of it: a new **The Data** page grounding the
catalog, a **Start Here** guided learning path, the nav reorganized into audience lanes
(Start Here → Learn / Reviewer Guide → Results / Reference), a **rebuilt Architecture
diagram** (compact top-down + a five-step prose walkthrough + a technique table), and a
site-wide prose pass that replaced roadmap-internal shorthand (`Phase N`, `Track B.N`,
`plan §N`, `rung`) with plain descriptive language. Site builds `--strict` clean.

**CI now gates code, not just docs.** A new `test.yml` runs `ruff` + `black --check`, a
**notebook render-freshness check** (`scripts/check_notebook_render_fresh.py` /
`make docs-notebooks-check`), and `pytest` on every push and PR. The suite runs on the
synthetic fixture under a lean `[dev]` install (192 passed / ~19 skipped — SBERT/torch/
matplotlib tests skip in CI, exercised locally with full extras), closing the prior gap
where nothing ran the tests in CI.

## Next task

Optional polish, in priority order: (1) **Stage 3** — a standalone RecSys concept primer
(much of it is already folded into *Start Here*, so this is genuinely optional); (2) the
**Pygments follow-up** (below); (3) decide **Docker-image hosting** (below).

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| ~~Pin the docs Pygments dep~~ | **Done** — `Pygments<2.19` added to `requirements-docs.txt` and the `[docs]` extra; verified via a from-scratch venv resolve (CI's exact path) → `mkdocs build --strict` passes. | Sandeep | resolved |
| **Docs dep drift (optional)** | `requirements-docs.txt` (ranges → material 9.7.6, pymdownx 10.21.3) and the `[docs]` extra (exact → material 9.5.49, pymdownx 10.14) resolve to *different* versions despite the "mirror" comment. Both build clean today. Reconcile to one pin style for fully reproducible builds if desired. | Sandeep | when convenient |
| ~~Node 20 deprecation in the workflows~~ | **Done** — bumped `checkout@v7`, `setup-python@v6`, `upload-pages-artifact@v5`, `deploy-pages@v5` (all node24) in `test.yml` + `docs.yml`. Local dev Node also moved off EOL v16 to LTS **v24.18.0** via nvm (`default -> lts/*`). | Sandeep | resolved |
| Where to host the Docker image | Cloud Run / HF Spaces (Docker) / leave as a local `docker run` portfolio artifact | Sandeep | when convenient |
| pyproject **import** package stays `courserec` | keep (hyphens illegal in module names) vs rename to `course_recommender_lab` (228 sites) | Sandeep | not planned |
| ~~Rename local working dir~~ / ~~Enable GitHub Pages~~ | **Both done** — dir renamed + `.venv` repaired; Pages live. Loose end: the `.claude` memory path still keys off the old dir name if you rely on it. | Sandeep | resolved |

## Blockers / waiting-on

None.

## First task for next session

Decide whether to build the Stage 3 concept primer (optional — *Start Here* already
covers the core idea). Either way, fold in the **Pygments pin** as a small, safe hardening
commit (pin in both `requirements-docs.txt` and the `[docs]` extra together). If
regenerating the notebook pages, run `make docs-notebooks` locally (needs the catalog +
optional Ollama) before pushing.
