# Convenience targets for the docs site (ADR-0015). The rest of the repo is driven
# by the scripts under scripts/ and pytest — see docs/RUNBOOK.md for the full set.
.PHONY: docs-notebooks docs-notebooks-check docs-build docs-serve

# Execute the teaching notebooks and write the site's pre-executed render artifacts
# to docs/notebooks/*.ipynb. Run locally where the catalog (+ optional Ollama for 08)
# exists: pip install -e ".[notebooks,semantic]". Pass NB="01 03" to render a subset.
docs-notebooks:
	python scripts/render_notebooks.py $(NB)

# Fail if any notebooks/NN_*.py source was committed after its docs/notebooks/*.ipynb
# render (a forgotten `make docs-notebooks`). Git-history based — needs full history.
docs-notebooks-check:
	python scripts/check_notebook_render_fresh.py

# Build the static site the way CI does (fails on broken links / nav). pip install -e ".[docs]"
docs-build:
	python -m mkdocs build --strict

# Live-reload preview at http://127.0.0.1:8000
docs-serve:
	python -m mkdocs serve
