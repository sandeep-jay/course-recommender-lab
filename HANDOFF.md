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

## Next task

Optional polish, in priority order: (1) **Stage 3** — a standalone RecSys concept primer
(much of it is already folded into *Start Here*, so this is genuinely optional); (2) the
**Pygments follow-up** (below); (3) decide **Docker-image hosting** (below).

## Open decisions

| Decision | Options | Owner | Due |
|---|---|---|---|
| **Pin the docs Pygments dep** | `requirements-docs.txt` + the `[docs]` extra are range-pinned; a fresh resolve pulls Pygments 2.20, which breaks pymdown-extensions 10.14 (`filename=None`). CI is green only via its build cache. Pin `Pygments<2.19` (known-good, tested locally) **or** bump `pymdown-extensions>=10.15` — but note: editing `requirements-docs.txt` invalidates the CI cache, so the pin must be correct in the same commit. | Sandeep | before the CI docs cache next evicts |
| **Node 20 deprecation in `docs.yml`** | `actions/checkout@v4`, `setup-python@v5`, `upload-artifact@v4` are being force-migrated to Node 24 by GitHub (warning only today). Bump the action pins when convenient. | Sandeep | when convenient |
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
