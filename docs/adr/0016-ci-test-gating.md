# ADR-0016: CI test gating with a lean install and a notebook render-freshness guard

**Date:** 2026-07-04
**Status:** Accepted

## Context
Until now the only CI workflow was `docs.yml` (build + deploy the site). **Nothing ran
the test suite or linters in CI** — the 211 tests only ran on a developer's machine. For
a portfolio that advertises "tests alongside implementation," an ungated suite is a
visible gap: a reviewer sees no green "tests passing" signal, and a regression could land
on `main` unnoticed.

Two constraints shape *how* a test workflow can run here:

1. **The real data and heavy deps are absent/expensive in CI.** The catalog CSV and
   `artifacts/` are gitignored (so absent on a cold clone), and the ML stack
   (`torch`, `sentence-transformers`, `faiss`, `matplotlib`, `umap`) is slow and heavy to
   install on a runner.
2. **The published notebook renders can drift.** [ADR-0015](0015-docs-site.md) keeps
   `notebooks/*.py` as source and commits the pre-executed `docs/notebooks/*.ipynb` as a
   **manual** `make docs-notebooks` artifact — so a `.py` edit can leave its render stale,
   and CI *cannot* re-execute notebooks to check (no catalog, no Ollama, by that ADR).

## Decision
1. **Add `.github/workflows/test.yml`** with a `lint` job (`ruff` + `black --check` + the
   render-freshness check below) and a `test` job (`pytest`), on every push to `main` and
   every PR.
2. **Run the suite under a lean `[dev]` install, not the full extras.** The tests use a
   tiny synthetic-catalog fixture (`tests/conftest.py`), and every heavy-dep or
   real-data test guards with `pytest.mark.skipif` / `importorskip` / a data-present
   skip. So under `[dev]` the SBERT/torch, matplotlib, and real-catalog tests **skip
   cleanly** (≈185 passed / 26 skipped in CI) instead of forcing a slow, flaky ML install.
   The heavy tests are exercised locally with the full extras. One unguarded matplotlib
   test was fixed to `importorskip` to match the existing pattern.
3. **Add `scripts/check_notebook_render_fresh.py`** (+ `make docs-notebooks-check`): a
   git-timestamp check that fails when a `notebooks/NN.py` was committed *after* its
   `docs/notebooks/NN.ipynb` render. It runs in the CI `lint` job with `fetch-depth: 0`,
   catching the one drift ADR-0015's manual re-render cannot auto-fix.

## Consequences
- **CI coverage is deliberately a subset.** SBERT/rerank/clustering-viz and real-catalog
  tests skip on the runner. Accepted: they run locally, the fixture suite covers the core
  logic and the `Recommender` interface contract, and CI stays fast (~30 s lint) and
  reliable (no multi-minute torch install to flake on).
- **The freshness guard is timestamp-based, not a content diff.** A no-op `.py` edit
  (a comment, a rename) trips it — the fix is a harmless `make docs-notebooks`. It also
  cannot detect a render that went stale *without* a `.py` edit (e.g. underlying results
  changed); that remains manual discipline, as flagged in ADR-0015.

## Alternatives rejected
- **Full-extras CI** (install `[dev,semantic,viz]`): the torch install is heavy, slow, and
  a flake surface, for little marginal coverage over a tiny synthetic fixture.
- **Execute-and-diff the notebooks in CI:** would need the catalog + a live Ollama — the
  exact infeasibility ADR-0015 was built around.

Builds on [ADR-0014](0014-teaching-notebooks.md) (notebooks) and
[ADR-0015](0015-docs-site.md) (docs site + manual render).
