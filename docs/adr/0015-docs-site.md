# ADR-0015: Documentation site — MkDocs Material on GitHub Pages, with pre-executed notebooks

**Date:** 2026-07-04
**Status:** Accepted

## Context
The repo is feature-complete (phases 0–8) and heavily documented — a rich README, a
RUNBOOK, RESULTS/TRADEOFFS, 14 ADRs, and ten teaching notebooks — but all of it is
readable only as raw Markdown on GitHub or as `.py` scripts that must be converted before
they render. As a portfolio piece it needs a **published, navigable site** with the same
shape as the sibling repos (`scribe-iq-lakehouse`, `campus-rag-assistant`), both of which
publish MkDocs Material to GitHub Pages.

Two questions shaped the decision:

1. **Toolchain and hosting.** What builds the site, and where does it live, given the
   portfolio already has a convention?
2. **How do the teaching notebooks appear?** They are the centerpiece of a *learning*
   lab, but [ADR-0014](0014-teaching-notebooks.md) deliberately keeps `.py` percent
   scripts as the versioned source and **gitignores** the `.ipynb`. A notebook rendered
   without outputs undersells it; executing notebooks *in CI* is infeasible here — the
   catalog CSV is gitignored (absent in CI), and notebook 08 needs a local Ollama daemon.

## Decision
1. **MkDocs Material → GitHub Pages, matching the portfolio.** `mkdocs.yml` (Material
   theme, `strict: true`, indigo palette, mermaid via SuperFences, `edit_uri`) plus a
   `.github/workflows/docs.yml` that builds `--strict` and deploys via the Pages-artifact
   action (no `gh-pages` branch). A `[docs]` extra in `pyproject.toml` mirrors
   `requirements-docs.txt` (CI installs from the latter for a fast, cache-keyed build).
2. **Reuse existing docs; add a thin reviewer-facing layer.** RESULTS, TRADEOFFS,
   RUNBOOK, the ADRs, and the roadmap wire straight into the nav. Five new pages serve
   the two audiences — `index`, `reviewer-guide` (hiring skim), `ARCHITECTURE`,
   `case-study` (learner narrative), `about`. `changelog.md` includes the repo-root
   `CHANGELOG.md` verbatim via `pymdownx.snippets` (single source of truth).
3. **Render notebooks with `mkdocs-jupyter`, from *pre-executed* `.ipynb` committed under
   `docs/notebooks/`.** `scripts/render_notebooks.py` (a `make docs-notebooks` target)
   executes each `notebooks/*.py` locally — where the catalog and Ollama exist — and
   writes the executed notebook to `docs/notebooks/NN_*.ipynb`. The plugin renders it with
   `execute: false`, so **CI never runs a notebook** and needs no data, models, or Ollama.
   This is a **scoped, deliberate exception to ADR-0014**: the `.py` percent script
   remains the source of truth; the `docs/notebooks/*.ipynb` are a *generated publish
   artifact*, never hand-edited. The `.gitignore` ban on committed notebooks stays, but is
   narrowed from all `*.ipynb` to `notebooks/*.ipynb`.

## Alternatives considered
- **Render the `.py` source with no outputs (`execute: false` over `notebooks/*.py`).**
  Fully CI-reproducible and no ADR-0014 exception, but a notebook page with no
  leaderboard, no plots, and no live-LLM responses is exactly the weak artifact the
  teaching notebooks exist to avoid. Rejected — outputs *are* the lesson here.
- **Execute notebooks in the CI docs build.** Rejected: the catalog CSV is gitignored
  (not in a CI checkout) and notebook 08 needs a local Ollama daemon; CI would be slow,
  flaky, and would still fail the data-dependent cells.
- **Link out to the `.py` scripts on GitHub, no in-site rendering.** Simplest, but the
  weakest experience for the learner audience — rejected.
- **A different SSG (Docusaurus, Sphinx, plain README).** Rejected for portfolio
  consistency: the two published sibling repos already standardize on MkDocs Material, so
  matching them is lower-friction and visually coherent.
- **`mkdocs gh-deploy` (gh-pages branch).** Rejected in favor of the Pages-artifact
  action — no long-lived generated branch, same as the sibling repos.

## Consequences
**Positive.** A published, navigable site at
`https://sandeep-jay.github.io/course-recommender-lab/` matching the portfolio, with the
teaching notebooks shown as real executed notebooks (SBERT rankings, the clustering map,
the live qwen3:8b responses in 08). `strict: true` fails the build on a broken internal
link, so the published site can't rot silently. Existing docs are reused, not rewritten.

**Honest caveats.** (1) The published notebook outputs are a **manual re-render**:
`make docs-notebooks` must be run when a notebook's code or the underlying results change,
or `docs/notebooks/*.ipynb` will lag the source (a `.py` edit alone won't update the
rendered page). This is the price of showing outputs without executing in CI. (2) The
`[docs]` and `[notebooks]` extras are **not co-installable** — `mkdocs-jupyter` pins
`ipykernel<7` while `[notebooks]` pins `7.3.0`; they are separate steps (render vs build)
that never share an env, and CI installs only `[docs]`. (3) The committed `.ipynb`
reintroduce some JSON diff noise under `docs/notebooks/` — bounded to the publish artifact,
not the source, and the metadata filter in `[tool.jupytext]` keeps it minimal.

**Neutral.** The distribution/display name was aligned to `course-recommender-lab` (repo
name); the Python import package stays `courserec` (hyphens are illegal in module names).
Builds on ADR-0014 (notebooks) and ADR-0012/0013 (the UI/Docker surfaces the site documents).

## Implementation notes
`mkdocs.yml`, `requirements-docs.txt`, `.github/workflows/docs.yml`, a `[docs]` extra in
`pyproject.toml`, and a `Makefile` (`docs-notebooks` / `docs-build` / `docs-serve`).
`scripts/render_notebooks.py` executes the notebooks with the kernel cwd set to
`notebooks/` (so `import nbtools` resolves) and writes to `docs/notebooks/`. New pages:
`docs/index.md`, `reviewer-guide.md`, `ARCHITECTURE.md`, `case-study.md`, `about.md`,
`changelog.md`, and `docs/notebooks/index.md`. `.gitignore` adds `site/` and narrows the
notebook ban to `notebooks/*.ipynb`. One-time maintainer step: Settings → Pages →
Source = "GitHub Actions".
