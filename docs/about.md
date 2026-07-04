# About

**course-recommender-lab** is a content-based recommender-systems study on the UC
Berkeley course catalog, built by **Sandeep Jayaprakash** as part of a portfolio of
production-pattern ML and data-engineering projects.

## Why a "lab"

The name is deliberate. This isn't a single model shipped once — it's a controlled
environment where eight technique families are held to **one interface**, scored by
**one evaluation harness**, and ranked on **one leaderboard**. The value is in the
*comparison* and the *methodology*: getting the measuring stick right, enforcing leakage
discipline, reporting confidence intervals, and treating negative results as first-class
findings rather than hiding them.

## Design principles

- **Honest documentation.** Limitations and failures are first-class. Three techniques
  that *should* have won and didn't are the most prominent results, not footnotes.
- **Production patterns from day one.** A swappable ABC, persisted artifacts, pinned
  dependencies, an ADR per architectural decision, tests alongside every technique.
- **Reproducible and key-free.** Global `RANDOM_SEED = 42`; the whole repo runs
  end-to-end with **no API key**. The LLM rung uses a local Ollama daemon and degrades
  gracefully when it's absent.
- **Portable.** No hardcoded paths, keys, or magic numbers; everything runs on a laptop.

## The rest of the portfolio

- **[scribe-iq](https://github.com/sandeep-jay/scribe-iq)** — a grounded clinical
  documentation AI prototype: synthetic Synthea patient spine, public clinical-note
  corpora, RAG over pgvector, FastAPI + Next.js, multi-cloud LLM providers, and governed
  audit workflows.
- **[scribe-iq-lakehouse](https://github.com/sandeep-jay/scribe-iq-lakehouse)** — a
  production-pattern healthcare data lakehouse: a Bronze→Silver→Gold medallion over
  Synthea Coherent FHIR, built *twice* (Polars + delta-rs locally, Spark + Delta on
  Microsoft Fabric), orchestrated with Dagster, emitting one governed Gold data contract.
- **[campus-rag-assistant](https://github.com/sandeep-jay/campus-rag-assistant)** — a
  multicloud RAG + agentic helpdesk platform: LangGraph orchestration, AWS Bedrock /
  Azure AI Search providers, cited answers, human-in-the-loop ticket filing, and RAGAS
  evals behind full CI/security gates.

## Source & license

- **Repository:** [github.com/sandeep-jay/course-recommender-lab](https://github.com/sandeep-jay/course-recommender-lab)
- **License:** MIT
- **Contact:** [github.com/sandeep-jay](https://github.com/sandeep-jay)
