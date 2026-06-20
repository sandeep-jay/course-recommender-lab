# ADR-0014: Teaching notebooks — reimplement-from-scratch over a tested library

**Date:** 2026-06-20
**Status:** Accepted

## Context
The repo has tried ~18 technique×config rungs across 8 families, all behind the
`Recommender` interface and scored by one harness. The code is clean and tested —
but it reads as *finished machinery*, not as a *learning path*. The project's
stated purpose is "a sandbox for **learning** recommender systems," and the value
of that learning is locked inside `fit()`/`recommend()` calls. We want notebooks
that make the journey legible: how each technique actually works, step by step,
with its evaluation shown live. Three questions shaped the decision:

1. **Thin (call the library) or reimplement (write the algorithm out)?** A thin
   notebook that calls `TfidfRecommender(...).fit()` hides exactly the mechanism a
   learner wants to see; a full parallel reimplementation risks a second codebase
   that drifts from the library and silently goes wrong.
2. **What granularity?** One mega-notebook, one per technique×config (~18), or one
   per family (8)?
3. **How do notebooks not rot?** `.ipynb` are JSON blobs (awful diffs) with hidden
   execution state; a committed notebook that no longer runs is a portfolio
   liability, not an asset.

## Decision
1. **Reimplement the technique from primitives; reuse the library only for
   plumbing.** Each technique notebook builds the method from numpy/scikit-learn
   building blocks so every step is visible — *that is the deliverable*. It reuses
   the library only for what isn't being taught: loading the catalog
   (`courserec.data`) and the evaluation harness (`courserec.eval`). The headline
   numbers therefore come from the *same* harness as the leaderboard, and each
   notebook **cross-checks** its from-scratch ranking against the library version
   (`nbtools.top_k_overlap`) — teaching divergence is allowed and expected, result
   divergence is caught. The library stays the source of truth for *numbers*; the
   from-scratch code is the source of *understanding*. Where a mechanism can't be
   honestly reimplemented (a transformer in 03, an LLM in 08) the notebook stays
   thin and teaches the *surrounding* machinery (caching, ANN, the prompt) instead.
2. **One notebook per technique family**, mapped 1:1 to `src/courserec/
   recommenders/*.py`, the roadmap phases, and the ADRs — bracketed by a foundation
   notebook (`00`, the shared data + eval the others assume) and a synthesis
   notebook (`09`, the cross-technique comparison). Configs (stopwords, n-grams,
   `k1`/`b`, fusion weight) become a **sweep inside** the family notebook — which is
   the actual teaching moment ("watch the knob move the metric"), not a reason to
   split into 18 files.
3. **Evaluate live, from warm artifacts.** Each notebook *runs* the harness on its
   technique (the reader sees the metrics generated, not pasted), but fits from the
   cached `artifacts/` so it executes in seconds, with a reduced bootstrap count.
4. **jupytext source + nbmake execution.** The committed source of each notebook is
   a **`.py` percent script** (clean git diffs, lintable); the `.ipynb` is generated
   by jupytext on demand and **gitignored**. `nbmake` executes every generated
   notebook (`pytest --nbmake`) so a cell that breaks fails loudly — the
   "can't silently rot" guarantee. Shared display helpers live in a small,
   unit-tested `notebooks/nbtools.py` (the one notebook file linted as library code).

## Alternatives considered
- **Thin notebooks over the library (the original instinct).** Rejected as the
  primary mode: it teaches nothing about the mechanism and reduces each technique
  to a one-line call. Kept only where reimplementation would mean faking a
  transformer/LLM.
- **Full reimplementation with no library tie-in.** Rejected: a second
  implementation of the metrics/eval would drift from `run_eval.py`, and the
  notebooks' numbers would stop matching the leaderboard. The plumbing-reuse +
  cross-check split keeps teaching freedom without numeric drift.
- **One notebook per technique×config (~18).** Rejected: duplicated setup and, worse,
  it splits apart the one thing worth teaching — how a config knob moves the metric.
  A sweep inside the family notebook shows it in one place.
- **Commit `.ipynb` with rendered outputs** (best GitHub preview). Rejected: noisy
  JSON diffs and no guarantee the notebook still runs. jupytext `.py` + on-demand
  `.ipynb` + nbmake trades the inline preview for clean history and an execution
  test; a reader regenerates the rendered notebook with one command.
- **`nbstripout` only.** Rejected: strips diff noise but gives no execution
  guarantee and no readable `.py` mirror.

## Consequences
**Positive.** The "we tried a lot of techniques" story becomes a readable,
runnable path: each family is a from-scratch walkthrough ending in its real eval,
and the foundation/synthesis bookends carry the shared harness and the head-to-head.
Notebooks can't silently break (nbmake), can't drift in their numbers (shared
harness + cross-check), and diff cleanly (jupytext `.py`). The library is unchanged
and remains the single source of truth.

**Honest caveats.** (1) GitHub won't render an output-rich notebook from the
committed `.py` — a viewer must `jupytext --to ipynb` first (documented in
`notebooks/README.md`). (2) "Live eval" depends on warm `artifacts/`; a cold clone
must run the pipeline first (same prerequisite as the rest of the repo). (3) The
numbered notebook scripts are excluded from `ruff`/`black` (mid-file imports,
display expressions are idiomatic in notebooks); only `nbtools.py` is linted.
(4) nbmake adds a slow, opt-in test lane — kept out of the default `pytest` run so
the 205-test suite stays fast.

**Neutral.** New optional `notebooks` extra (jupytext, nbmake, ipykernel,
matplotlib); absent it, the library, eval, UI, and Docker image are unaffected.
Builds on every prior ADR (each notebook cites the ADR for its technique).

## Implementation notes
`notebooks/` holds `00`–`09` as jupytext `.py` percent scripts (this PR ships
`00_data_and_eval` + `01_lexical` as the pattern; `02`–`09` follow), `nbtools.py`
(`recs_to_frame`, `top_k_overlap`, `plot_metric_ci` — tested in
`tests/test_nbtools.py`), and a `README.md` index. `pyproject` adds the `notebooks`
extra, a `[tool.jupytext]` metadata filter, and excludes the numbered scripts from
`ruff`/`black`. `.gitignore` drops `notebooks/*.ipynb`. RUNBOOK + README document
the generate-then-nbmake workflow.
