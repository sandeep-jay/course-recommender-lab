# Engineering Case Study

*The full arc — the problem, the constraint that shaped it, the eight techniques, and
what the results actually taught. If the [Reviewer Guide](reviewer-guide.md) is the
trailer, this is the film.*

## The problem

The UC Berkeley course catalog has ~11,000 courses and *no* record of who took what.
There are no clicks, no ratings, no enrollments — nothing a collaborative filter could
learn from. So the question narrows to a pure **content-based** one:

> Given a seed course (or a free-text sentence), which other courses are most similar —
> judged from **text and metadata alone**?

That missing-interaction-data constraint is the whole design premise. It puts every
technique on equal footing: none of them can lean on behavioral signal, so the
comparison is a clean test of *how well each represents course content*.

## The measuring stick came first

Before any technique, the [evaluation harness](ARCHITECTURE.md#the-evaluation-harness).
With no ground-truth "these are similar" labels, we needed a proxy — and the catalog
provides one almost for free: **cross-listed courses**. When the same course is offered
under two departments (say a course listed under both STATS and COMPSCI), those listings
are, by definition, near-identical twins. A good recommender should rank a course's
twin near the top.

That gives an automatic primary lens. But it comes with a sharp caveat that shapes every
later conclusion: **twin descriptions are often near-identical text**, so recovering them
is easy for *any* method that reads words. Cross-listing validates *correctness* more
than *quality*. So two more lenses back it up:

- **Same-subject coherence** — a weak sanity floor, never optimized for (a
  same-subject-only model would ace it while being useless).
- **A hand-labeled judged-query set** — 44 deliberately paraphrase-extreme free-text
  queries, the *only* way to measure `recommend_by_text`, where the query shares few or
  no literal words with the target. This is where representation quality actually shows.

Every headline number carries a **bootstrap confidence interval**, and no winner is
declared on a sub-CI gap. Getting the ruler right before building the thing being
measured is the single most important decision in the project
([ADR-0002](adr/0002-eval-harness-design.md)).

## Eight techniques, one interface

Each technique family is a chapter — and each has a [teaching notebook](notebooks/index.md)
that builds it from primitives on the real catalog:

1. **Lexical** (TF-IDF, Okapi BM25) — the bag-of-words baseline. Strong on cross-listing
   *because* twins share vocabulary; the honest baseline everything else must beat.
2. **Topic models** (LSA, NMF, LDA) — compress the term space to latent topics. NMF is
   the surprise over-performer among the classical methods.
3. **Semantic vectors** (SBERT: MiniLM, MPNet) — dense sentence embeddings that capture
   *meaning*, not just shared words. This is the one that pulls ahead on free text.
4. **Retrieve → rerank → MMR** — SBERT retrieves, a cross-encoder reranks the top
   candidates, and an MMR knob trades relevance for diversity
   ([ADR-0005](adr/0005-rerank-mmr.md)).
5. **Course graph (personalized PageRank)** — a graph over cross-listing + subject/dept
   nodes, evaluated only on a **held-out edge split** so it never sees its own answers
   ([ADR-0006](adr/0006-graph-heldout.md)).
6. **Metadata fusion** — one-hot subject/department/level/units fused with TF-IDF under a
   weight λ ([ADR-0008](adr/0008-metadata-fusion.md)).
7. **Clustering + 2-D map** — KMeans / Ward / HDBSCAN over the SBERT vectors, a
   *diagnostic* of the embedding space, not a ranker ([ADR-0007](adr/0007-clustering-diagnostic.md)).
8. **LLM** (local qwen3:8b via Ollama) — three roles: extract tags, zero-shot rerank,
   and explain ([ADR-0009](adr/0009-llm-enrichment-ollama.md),
   [ADR-0010](adr/0010-llm-reranker.md), [ADR-0011](adr/0011-llm-explainer.md)).

## What the results taught

**SBERT MiniLM wins — and the *where* is the lesson.** On cross-listing twins it tops the
board (NDCG@10 0.971), but lexical methods are within its CI: when twin text is nearly
identical, meaning-vs-words barely matters. On **free-text queries** the story flips —
SBERT scores NDCG@10 **0.682 vs 0.499** for the best lexical config, with
non-overlapping CIs. **Semantic representation earns its keep exactly where the words
stop matching.** That single contrast is the project's thesis.

Then the three results that went against the obvious hypothesis — the ones worth the lab:

### 1. Metadata *hurts*

The intuition "subject and level must help find similar courses" is wrong for this
target. **99.7% of cross-listed twins span different subjects**, so one-hot subject/dept
features pull the *wrong* courses together — the more metadata weight, the lower the
cross-listing score. A plausible idea, cleanly ablated and falsified
([ADR-0008](adr/0008-metadata-fusion.md)).

### 2. The graph has no headroom

A personalized-PageRank graph recovers only **~23%** of held-out twins. Not because the
graph is bad — because the twins' *text* is already nearly identical, so a content method
has captured the signal before the graph adds anything. The structure has nothing left
to contribute ([ADR-0006](adr/0006-graph-heldout.md)).

### 3. The LLM loses at ranking, wins at explaining

The most instructive arc. The **tag rung** — distill each course to ~6–12 LLM-extracted
tags — *looked* like it beat every lexical baseline… on a 12.5% eval-target subset. Then
**full-catalog (100%) enrichment overturned it**: at full coverage it merely ties lexical
on cross-listing and falls *below* plain TF-IDF on free text, because distilling a
description to a handful of tags loses more signal than the LLM's normalization adds. A
textbook lesson in **evaluation-subset bias**. The **zero-shot reranker** then also failed
to beat SBERT — its top-20 is already near-ceiling, so reordering has no room and costs
~4 s/query. The productive move was to stop asking the LLM to *rank* and let it *explain*:
a one-line "why this fits" justification for an SBERT recommendation — where it clearly
pays for itself ([ADR-0009](adr/0009-llm-enrichment-ollama.md),
[ADR-0010](adr/0010-llm-reranker.md), [ADR-0011](adr/0011-llm-explainer.md)).

## What I'd do next

- A larger, more adversarial judged-query set — 44 queries is enough to separate SBERT
  from lexical but thin for finer distinctions; the CIs say so.
- Hard-negative mining for the cross-encoder reranker, to test whether rerank headroom
  exists on a *harder* candidate set than SBERT's near-ceiling top-20.
- Hybrid lexical+dense fusion (RRF), the one obvious combination not yet ablated.

## Read on

- The numbers in full: [Results & Findings](RESULTS.md), [Technique Tradeoffs](TRADEOFFS.md).
- The techniques hands-on: [Teaching Notebooks](notebooks/index.md).
- The decisions behind each: [ADR index](adr/README.md).
