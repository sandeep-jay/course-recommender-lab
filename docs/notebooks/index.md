# Teaching Notebooks

Ten step-by-step breakdowns — one per technique family, plus a data/eval foundation and
a cross-technique synthesis. Each notebook **builds the method from primitives on the
real catalog** and runs its evaluation live, so you see not just how a technique works
but how it *scores* ([ADR-0014](../adr/0014-teaching-notebooks.md)).

!!! info "These pages show real, executed outputs"
    Every notebook here is rendered from a **pre-executed** `.ipynb` — the tables,
    plots, rankings, and (for notebook 08) the live local-LLM responses are the actual
    outputs from running against the 11,073-course catalog. The versioned *source* is
    the `notebooks/*.py` jupytext percent script; these executed notebooks are the
    published render artifact ([ADR-0015](../adr/0015-docs-site.md)).

## The sequence

Notebook **00** is the foundation — every technique notebook assumes it. After that, the
numbered notebooks are independent; read the one whose technique interests you.

| # | Notebook | What it builds |
|---|---|---|
| 00 | [Data & Eval foundation](00_data_and_eval.ipynb) | The cleaned catalog and the three-lens evaluation harness — the shared plumbing every other notebook uses. |
| 01 | [Lexical](01_lexical.ipynb) | TF-IDF + cosine and Okapi BM25 from the term-document matrix up. The honest baseline. |
| 02 | [Topic models](02_topics.ipynb) | LSA, NMF, and LDA — compressing the term space to latent topics. |
| 03 | [Semantic vectors](03_embeddings.ipynb) | SBERT sentence embeddings (MiniLM, MPNet) — the winner on free text. |
| 04 | [Retrieve → rerank → MMR](04_rerank.ipynb) | A cross-encoder reranker over SBERT candidates, with an MMR diversity knob. |
| 05 | [Metadata fusion](05_metadata.ipynb) | Fusing one-hot facets with text — and why it *hurts* the cross-listing target. |
| 06 | [Course graph (PPR)](06_graph.ipynb) | Personalized PageRank on a held-out edge split — the leak-safe graph. |
| 07 | [Clustering & 2-D map](07_clustering.ipynb) | KMeans / Ward / HDBSCAN over the embeddings as a diagnostic, not a ranker. |
| 08 | [LLM enrichment & rerank](08_llm.ipynb) | Local qwen3:8b for tags, zero-shot rerank, and explanations — where the LLM loses and where it wins. |
| 09 | [Leaderboard synthesis](09_leaderboard.ipynb) | All techniques side by side — the cross-technique payoff. |

## Running them yourself

The notebooks live in `notebooks/` as jupytext `py:percent` scripts. To run locally:

```bash
pip install -e ".[notebooks,semantic]"
jupytext --to ipynb notebooks/01_lexical.py
jupyter lab notebooks/01_lexical.ipynb
```

Notebook 08 additionally wants a local [Ollama](https://ollama.com) daemon with
`qwen3:8b` pulled — but it degrades gracefully to its cache when Ollama is absent. See
the [Runbook](../RUNBOOK.md) for the full install matrix.
