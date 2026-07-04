# Reviewer Guide

*Read this in five minutes. It's written for someone skimming a portfolio — what this
project demonstrates, the results that matter, and exactly where to look.*

## What this is

A content-based recommender-systems study on the **UC Berkeley course catalog**
(~11,073 courses after cleaning). Eight technique families are implemented against
**one interface**, scored by **one evaluation harness** on **three lenses**, and ranked
on a **one-command leaderboard**. There is no user-interaction data, so collaborative
filtering is out of scope — every technique earns its ranking from text and metadata
alone.

## What it demonstrates

| Skill | Where to see it |
|---|---|
| **Breadth of RecSys / IR technique** | Lexical (TF-IDF, BM25), topic models (LSA, NMF, LDA), semantic embeddings (SBERT), learned cross-encoder rerank + MMR, a personalized-PageRank graph, metadata fusion, and three LLM approaches — [Results](RESULTS.md) |
| **Evaluation rigor** | Three lenses, five ranking metrics at k∈{5,10,20}, **bootstrap confidence intervals** on the primary metric, leakage discipline — [Architecture](ARCHITECTURE.md#the-evaluation-harness) |
| **Honest engineering judgment** | Three *negative* results documented as first-class findings, not buried — [below](#the-three-findings-that-matter) |
| **Production patterns** | Swappable ABC, persisted artifacts, pinned deps, one ADR per decision (15), tests alongside every technique — [Architecture](ARCHITECTURE.md) |
| **Shipping** | A live Streamlit UI, a warm offline Docker image, and ten executed teaching notebooks — [Operations](RUNBOOK.md), [Notebooks](notebooks/index.md) |

## The headline result

**SBERT MiniLM tops both ranking lenses** — and on free-text queries it beats the best
lexical configuration *decisively*:

- **Free text:** NDCG@10 **0.682** vs **0.499** (best lexical) — non-overlapping CIs.
- **Cross-listing twins:** NDCG@10 **0.971** — but here lexical methods are within CIs,
  because near-identical twin text makes the task easy. We never crown a winner on a
  sub-CI gap.

## The three findings that matter

The point of a lab is what you *learn*, and the most instructive results here are the
ones that contradicted the obvious hypothesis:

1. **Metadata fusion makes it worse.** Fusing subject/department/level/units with text
   *lowers* cross-listing NDCG@10 — because **99.7% of cross-listed twins span different
   subjects**, so metadata pulls the wrong courses together. A clean, documented
   ablation of a plausible-but-wrong idea. → [ADR-0008](adr/0008-metadata-fusion.md)

2. **The graph can't beat content.** A personalized-PageRank model over cross-listing +
   subject/dept nodes, evaluated only on a **held-out edge split** (the one technique
   allowed to touch the ground-truth column, and only under a leak-safe split),
   recovers just **~23%** of held-out twins — because the twin *text* is already nearly
   identical, leaving the graph no headroom. → [ADR-0006](adr/0006-graph-heldout.md)

3. **The LLM loses at ranking, wins at explaining.** Two LLM ranking approaches — tag
   distillation and zero-shot rerank — both **failed to beat plain SBERT** (the tag
   approach *looked* like a winner on a 12.5% subset, then **full-catalog enrichment overturned
   it**). The honest conclusion: a local qwen3:8b earns its cost *justifying* a
   recommendation, not producing one. → [ADR-0009](adr/0009-llm-enrichment-ollama.md),
   [ADR-0010](adr/0010-llm-reranker.md), [ADR-0011](adr/0011-llm-explainer.md)

## Where to look (by interest)

- **"Show me the results."** → [Results & Findings](RESULTS.md) and
  [Technique Tradeoffs](TRADEOFFS.md).
- **"Show me the design."** → [Architecture](ARCHITECTURE.md) and the
  [ADR index](adr/README.md).
- **"Show me how a technique actually works."** → the
  [Teaching Notebooks](notebooks/index.md) build each one from primitives on the real
  catalog and run its eval live.
- **"Show me it runs."** → the [Runbook](RUNBOOK.md) — every script, flag, and install
  tier, end to end with no API key.

## Ground rules the code enforces

Three project invariants show up everywhere and are worth knowing while reading:

- **No leakage.** `Cross-Listed Course(s)` is the evaluation ground truth; no technique
  reads it as a feature (the graph is the sole exception, under a held-out split).
- **The `"-"` null token.** The catalog uses the string `"-"` for missing data; it's
  replaced with real NA on load and never treated as a value.
- **Reproducible, key-free.** Global `RANDOM_SEED = 42`; the whole repo runs end-to-end
  with **no API key** (the LLM technique uses a local Ollama daemon and degrades gracefully
  when it's absent).
