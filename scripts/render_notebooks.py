"""Render the teaching notebooks to executed ``.ipynb`` for the docs site (ADR-0015).

The versioned notebook *source* is the ``notebooks/*.py`` jupytext percent scripts
(ADR-0014). The published MkDocs site, however, needs notebook pages *with outputs*
so a reviewer sees the real leaderboard, plots, and rankings — not bare code. This
script bridges the two: it reads each ``notebooks/NN_*.py``, executes it against the
real catalog (and a live Ollama daemon for ``08``, degrading to cache when absent),
and writes the executed notebook to ``docs/notebooks/NN_*.ipynb``, which *is*
committed as the site's render artifact (mkdocs-jupyter renders it with
``execute: false`` — CI never runs a notebook).

Run it locally where the data + optional Ollama exist:

    pip install -e ".[notebooks,semantic]"
    python scripts/render_notebooks.py            # all notebooks
    python scripts/render_notebooks.py 01 03      # only 01_* and 03_*

Kernel working directory is set to ``notebooks/`` so ``import nbtools`` resolves;
catalog paths are absolute (``courserec.config``) so they resolve regardless.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("render_notebooks")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "notebooks"
# Per-cell execution ceiling — the SBERT/LLM cells are the slow ones. Generous so a
# cold model load or a live qwen3:8b call never trips a spurious timeout.
CELL_TIMEOUT_S = 1200


def source_notebooks(selectors: list[str]) -> list[Path]:
    """Return the numbered notebook source scripts to render, sorted.

    Args:
        selectors: Optional numeric prefixes (e.g. ``["01", "03"]``). Empty means
            all ``notebooks/NN_*.py`` scripts.

    Returns:
        Matching ``notebooks/*.py`` paths in filename order.
    """
    scripts = sorted(p for p in NOTEBOOKS_DIR.glob("[0-9]*.py"))
    if not selectors:
        return scripts
    return [p for p in scripts if any(p.name.startswith(s) for s in selectors)]


def strip_stderr_streams(notebook: nbformat.NotebookNode) -> None:
    """Drop ``stderr`` stream outputs from every cell, in place.

    Batch execution emits library warnings (tqdm's missing-widget notice, sklearn
    matmul RuntimeWarnings) as ``stderr`` streams. They are pure noise on the
    published site and — worse — embed the **absolute local path** of the executing
    machine (the developer's home directory) in the committed notebook. Removing the
    ``stderr`` streams cleans both, while keeping ``stdout`` and rich display outputs
    (tables, plots) untouched.

    Args:
        notebook: An executed notebook node; mutated in place.
    """
    for cell in notebook.cells:
        outputs = cell.get("outputs")
        if not outputs:
            continue
        cell["outputs"] = [
            o
            for o in outputs
            if not (o.get("output_type") == "stream" and o.get("name") == "stderr")
        ]


def render_one(script: Path) -> Path:
    """Execute one notebook source script and write the executed ``.ipynb``.

    Args:
        script: A ``notebooks/NN_*.py`` jupytext percent source file.

    Returns:
        The path of the written executed notebook under ``docs/notebooks/``.
    """
    logger.info("executing %s ...", script.name)
    notebook = jupytext.read(script)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_S,
        kernel_name="python3",
        # Run with cwd = notebooks/ so the notebooks' ``import nbtools`` resolves.
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    strip_stderr_streams(notebook)
    out_path = OUTPUT_DIR / f"{script.stem}.ipynb"
    nbformat.write(notebook, out_path)
    logger.info("wrote %s", out_path.relative_to(PROJECT_ROOT))
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Render the selected notebooks; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selectors",
        nargs="*",
        help="Numeric prefixes to render (e.g. 01 03); default: all.",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts = source_notebooks(args.selectors)
    if not scripts:
        logger.error("no matching notebooks for %s", args.selectors or "<all>")
        return 1

    for script in scripts:
        render_one(script)
    logger.info("rendered %d notebook(s) to %s", len(scripts), OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
