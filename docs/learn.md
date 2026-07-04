# Start Here — learn recommender systems from a real catalog

**New to recommender systems, or just want the guided tour?** You're in the right place.
This lab builds eight families of content-based recommenders from the ground up on
11,073 real UC Berkeley courses — and this page is the single ordered path through them,
from the core idea to a ranked leaderboard.

!!! tip "If you only have a few minutes"
    Read **[The core idea](#the-core-idea)** below, skim **[The Data](data.md)**, then
    open **[Notebook 03 — SBERT](notebooks/03_embeddings.ipynb)** to see the winning
    technique built and scored. That's the whole story in miniature. Come back for the
    rest when you have an afternoon.

## The core idea

A **recommender system** answers: *given one thing, what other things are most like it?*
There are two classic ways to answer:

- **Collaborative filtering** — "people who liked this also liked that." It needs
  *interaction data* (clicks, ratings, purchases).
- **Content-based** — "this thing is similar to that thing *because of what they are*."
  It needs only the items' own text and attributes.

The course catalog has **no clicks and no ratings** — nobody's interaction history is in
it. So collaborative filtering is impossible here, and this whole lab is a focused study
of the **content-based** approach: every technique must decide that two courses are
similar using only their title, description, and metadata.

That leaves one question, asked eight different ways: *given a course (or a sentence of
free text), which other courses are the best matches?* The techniques range from simple
word-counting to neural sentence embeddings to a local LLM — and, refreshingly, the
simple ones are hard to beat.

## The path

Follow these in order the first time. Step&nbsp;1 is the foundation everything else
builds on; after that each technique notebook stands on its own.

<div class="grid cards" markdown>

- :material-numeric-1-circle: **Understand the raw material**

    Read **[The Data](data.md)** — what a course record looks like, and the
    *cross-listing ground truth* that lets us score recommendations without anyone
    hand-labeling "these two are similar."

- :material-numeric-2-circle: **See how techniques are scored**

    **[Notebook 00 — Data & Eval foundation](notebooks/00_data_and_eval.ipynb)** builds
    the cleaned catalog and the evaluation harness every later notebook reuses. Knowing
    *how we measure* is what makes the comparisons trustworthy.

- :material-numeric-3-circle: **Build the techniques, simple → smart**

    Work up the ladder, one notebook each. Each builds the method from primitives on the
    real catalog and scores it live (see the table below).

- :material-numeric-4-circle: **See them all ranked**

    **[Notebook 09 — Leaderboard](notebooks/09_leaderboard.ipynb)** and the
    **[Results & Findings](RESULTS.md)** put every technique side by side — including the
    three that *should* have won and didn't.

- :material-numeric-5-circle: **Explore it interactively**

    The same techniques power a Streamlit app (Explore, Compare, Leaderboard, and a 2-D
    map). See the **[Runbook](RUNBOOK.md)** to run the UI or the ready-made Docker image.

</div>

## The techniques, in learning order

Each row builds on ideas from the ones above it — simplest first, so you can feel *why*
each next step exists.

| Step | Technique | The idea, in one line | Notebook |
|---|---|---|---|
| 1 | **Lexical** (TF-IDF, BM25) | Count shared words, cleverly weighted. The baseline everything else must beat. | [01](notebooks/01_lexical.ipynb) |
| 2 | **Topic models** (LSA, NMF, LDA) | Compress thousands of words into a few dozen latent "topics." | [02](notebooks/02_topics.ipynb) |
| 3 | **SBERT** (sentence embeddings) | Let a neural model place each course in meaning-space. **The overall winner.** | [03](notebooks/03_embeddings.ipynb) |
| 4 | **Rerank + MMR** | Re-score the top candidates with a heavier model, and add a diversity dial. | [04](notebooks/04_rerank.ipynb) |
| 5 | **Metadata fusion** | Add subject/level/units to the text — and watch it *hurt*. A real lesson. | [05](notebooks/05_metadata.ipynb) |
| 6 | **Graph** (Personalized PageRank) | Treat cross-listings as a graph and walk it — evaluated leak-safely. | [06](notebooks/06_graph.ipynb) |
| 7 | **Clustering & 2-D map** | Not a ranker — a way to *see* the catalog's structure as a diagnostic. | [07](notebooks/07_clustering.ipynb) |
| 8 | **LLM** (local Ollama) | Tags, zero-shot reranking, and plain-English explanations. Where the LLM loses, and where it earns its keep. | [08](notebooks/08_llm.ipynb) |

## What you'll take away

- **How content-based recommenders actually work** — from word counts to embeddings to LLMs.
- **How to *evaluate* a recommender honestly** — three lenses, confidence intervals, and
  why you never crown a winner on a gap smaller than the noise.
- **That simpler is often better** — the headline finding is that a small, fast embedding
  model beats a local LLM at ranking, and that added metadata can make results *worse*.

---

**Ready?** Start with **[The Data](data.md)**, or jump straight to
**[Notebook 00](notebooks/00_data_and_eval.ipynb)**. Reviewing rather than learning?
The **[Reviewer Guide](reviewer-guide.md)** is the five-minute skim.
