# course-recommender-lab

**A sandbox for learning content-based recommender systems — built like production.**

Eleven thousand UC Berkeley courses, no clicks and no ratings, and one question:
*given a course (or a sentence), which other courses are most like it?* This repo
answers it eight different ways — lexical, topic-model, semantic-embedding, learned
rerank, graph, metadata-fusion, and LLM — holds every technique to **one interface**,
scores them all through **one evaluation harness**, and ranks them on **one
leaderboard**. The interesting results are the honest ones: three techniques that
*should* have won and didn't.

!!! tip "Two ways in"
    **Here to learn?** Follow the guided path in **[Start Here](learn.md)** — the core
    idea, the data, then each technique built from scratch. **Here to evaluate the work?**
    The **[Reviewer Guide](reviewer-guide.md)** is the five-minute skim.

<div class="grid cards" markdown>

- :material-flash-outline: **[Reviewer Guide](reviewer-guide.md)**
  The 5-minute tour — what this demonstrates, the headline results, where to look.

- :material-database-outline: **[The Data](data.md)**
  The UC Berkeley catalog itself — 11k courses, the schema, and the cross-listing ground truth.

- :material-sitemap-outline: **[Architecture](ARCHITECTURE.md)**
  The one contract every technique implements and the three-lens eval that scores them.

- :material-chart-box-outline: **[Results & Findings](RESULTS.md)**
  The leaderboard, the bootstrap CIs, and the three negative results that earned their place.

- :material-notebook-outline: **[Teaching Notebooks](notebooks/index.md)**
  Ten step-by-step breakdowns — each technique built from primitives on the real catalog.

</div>

## Why this exists

No user-interaction data exists for the catalog, so **collaborative filtering is out
of scope by construction** — this is a pure content-based study. That constraint is a
feature: it forces every technique to earn its ranking from *text and metadata alone*,
and it makes the comparison clean.

The project is deliberately built to production standards — a swappable interface,
persisted artifacts, pinned dependencies, an ADR per decision, tests alongside every
technique, and a one-command regenerable leaderboard — so it reads as engineering, not
a notebook dump.

## The headline

**SBERT (MiniLM) wins**, and it wins *decisively* on free-text queries where it matters:

| Lens | Winner | Margin |
|---|---|---|
| Cross-listing twins (item-to-item) | SBERT MiniLM · NDCG@10 **0.971** | Narrow — lexical methods are within CIs |
| Free-text queries (`recommend_by_text`) | SBERT MiniLM · NDCG@10 **0.682** | Decisive — vs 0.499 best lexical, non-overlapping CIs |

But the findings that make the lab worth reading are the ones that went the *other* way:

- **Metadata fusion *hurts*** the cross-listing target — 99.7% of twins span subjects,
  so subject/dept features actively mislead ([ADR-0008](adr/0008-metadata-fusion.md)).
- **The graph recovers only ~23%** of held-out twins — near-identical twin text already
  hands content methods the answer ([ADR-0006](adr/0006-graph-heldout.md)).
- **Two LLM approaches lost on ranking** — tag-distillation and zero-shot rerank both
  failed to beat plain SBERT; the local LLM earns its keep *explaining* a ranking, not
  producing one ([ADR-0009](adr/0009-llm-enrichment-ollama.md),
  [ADR-0010](adr/0010-llm-reranker.md), [ADR-0011](adr/0011-llm-explainer.md)).

See the [Case Study](case-study.md) for the full arc, or jump to the
[Results](RESULTS.md) and [Tradeoffs](TRADEOFFS.md).

## Part of a portfolio

course-recommender-lab is one of several production-pattern ML/data projects:

- **[scribe-iq](https://github.com/sandeep-jay/scribe-iq)** — grounded clinical
  documentation AI (RAG, pgvector, FastAPI, multi-cloud LLM providers).
- **[scribe-iq-lakehouse](https://github.com/sandeep-jay/scribe-iq-lakehouse)** — a
  Bronze→Silver→Gold healthcare lakehouse built twice (Polars/delta-rs and Spark/Fabric).
- **[campus-rag-assistant](https://github.com/sandeep-jay/campus-rag-assistant)** — a
  multicloud RAG + agentic helpdesk platform (LangGraph, RAGAS evals, HITL).

The [About](about.md) page has the full context.
