# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 · Lexical baselines — TF-IDF and BM25, from scratch
#
# > Assumes [notebook 00](00_data_and_eval.py) — the catalog, the leakage rule,
# > and the eval harness.
#
# The simplest rung of the similarity ladder: **bag-of-words** rankers that score
# courses by raw vocabulary overlap. No training, no embeddings — just "which
# courses use the same words as the seed." They are also the **correctness floor**
# for everything after: cross-listed twins share near-identical text, so a working
# lexical model *must* nail the cross-listing lens, or something is wrong.
#
# **How this notebook works (the house style):**
# 1. **Build the technique from primitives** so every step is visible.
# 2. **Cross-check** our from-scratch ranking against the library's
#    `TfidfRecommender` — they should agree.
# 3. **Sweep the knobs** (stopwords, n-grams, title weight) and watch the metric move.
# 4. **Score it live** through the real harness from notebook 00.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 60)
RANDOM_SEED = 42

courses = load_processed()
seed = "COMPSCI 189"  # "Introduction to Machine Learning"
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. TF-IDF + cosine, step by step
#
# **The idea.** Represent each course as a vector over the vocabulary, where each
# term is weighted by **TF-IDF** = (how often it appears in *this* course) ×
# (how *rare* it is across the catalog). Rare, on-topic words ("manifold",
# "phylogenetics") get high weight; ubiquitous words ("course", "students") get
# low weight. Two courses are similar if their vectors point the same way —
# **cosine similarity**, which for L2-normalized vectors is just a dot product.
#
# ### Step 1a — vectorize the corpus
# `TfidfVectorizer` does the TF-IDF weighting and L2-normalizes each row for us.
# The *matrix* it returns, and the cosine ranking we build on it, are the lesson.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

docs = courses["text"].fillna("").tolist()  # sparse-text rows already fell back to title
vec = TfidfVectorizer(stop_words="english")  # drop "the", "of", "and", …
X = vec.fit_transform(docs)  # (n_courses × n_terms) sparse, L2-normalized rows
course_ids = list(courses.index)
row_of = {cid: i for i, cid in enumerate(course_ids)}
print(f"TF-IDF matrix: {X.shape[0]:,} courses × {X.shape[1]:,} terms (sparse)")

# %% [markdown]
# ### Step 1b — what does the seed's vector actually contain?
# A vector is only as good as its heaviest terms. These are the words that will
# *drive* the seed's recommendations:

# %%
terms = np.array(vec.get_feature_names_out())
seed_vec = X[row_of[seed]]
order = np.argsort(seed_vec.toarray().ravel())[::-1][:10]
print(f"Top TF-IDF terms for {seed}:")
for j in order:
    print(f"   {terms[j]:20s}  {seed_vec[0, j]:.3f}")

# %% [markdown]
# ### Step 1c — rank by cosine = one sparse mat-vec
# Because rows are L2-normalized, `X @ seed_vecᵀ` *is* the cosine similarity of
# every course to the seed. We then drop the seed itself (the interface forbids a
# course recommending itself) and take the top-k.

# %%
def recommend_tfidf_from_scratch(seed_id: str, k: int = 10) -> list[tuple[str, float]]:
    """Cosine-rank the catalog against one seed, excluding the seed (teaching version)."""
    sims = (X @ X[row_of[seed_id]].T).toarray().ravel()  # cosine for every course
    sims[row_of[seed_id]] = -np.inf  # never recommend the seed itself
    top = np.argsort(sims)[::-1][:k]
    return [(course_ids[i], float(sims[i])) for i in top]


scratch = recommend_tfidf_from_scratch(seed, k=10)
pd.DataFrame(
    [(i + 1, cid, courses.loc[cid, "title"], round(s, 3)) for i, (cid, s) in enumerate(scratch)],
    columns=["rank", "course_id", "title", "cosine"],
)

# %% [markdown]
# The top hits are statistics / ML theory courses — exactly the neighborhood of an
# intro-ML course. And `COMPSCI C281A` (a cross-listing of 189's territory) shows
# why the cross-listing lens is *easy* for lexical methods: shared text → high cosine.

# %% [markdown]
# ## 2. Cross-check: does the library agree?
#
# The repo's `TfidfRecommender` is the same idea, packaged behind the
# `Recommender` interface with artifact caching. If our from-scratch ranking and
# the library's disagree, one of us is wrong. We measure agreement with
# `top_k_overlap` from `nbtools`.

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.lexical import TfidfRecommender

lib = TfidfRecommender(stopwords=True, ngram_max=1, title_weight=1)
lib.fit(courses)  # loads the cached artifact if present (warm); else fits once
lib_recs = lib.recommend_similar(seed, k=10)

overlap = top_k_overlap([c for c, _ in scratch], [r.course_id for r in lib_recs], k=10)
print(f"top-10 overlap (from-scratch vs library): {overlap:.0%}")
recs_to_frame(lib_recs, courses)

# %% [markdown]
# Expect roughly **70% top-10 overlap, with the very top results identical** —
# "largely agree," not bit-for-bit. Two honest reasons they diverge in the tail:
# the library repeats the *title* `title_weight` times (we used the raw `text`),
# and it reconstructs the seed's query from the fitted vocabulary via
# `inverse_transform` (dropping term-frequency), whereas we dotted the seed's full
# TF-IDF row. Same idea, two implementations, the same *neighborhood* — which is
# what a recommender is judged on. The differences live exactly where this lens
# can't see them (the tail), which is the cliffhanger for notebook 03.

# %% [markdown]
# ## 3. BM25 — saturating term frequency
#
# TF-IDF's weakness: a course mentioning "neural" 8 times looks 8× as "about"
# neural nets as one mentioning it once — but relevance saturates. **Okapi BM25**
# fixes this with two knobs:
#
# $$\text{score}(d,q)=\sum_{t\in q}\text{idf}(t)\cdot\frac{tf(t,d)\,(k_1+1)}{tf(t,d)+k_1\,(1-b+b\,|d|/\text{avgdl})}$$
#
# - **`k1`** caps how much repeated terms help (term-frequency *saturation*).
# - **`b`** normalizes by document length, so a long syllabus isn't unfairly
#   favored just for containing more words.
#
# The repo folds the whole per-document factor into a sparse weight matrix `W`, so
# scoring is *again* one mat-vec — `W @ q` with `q` a binary query-presence vector.
# That function **is** the formula above; let's use it on the seed's terms.

# %%
from sklearn.feature_extraction.text import CountVectorizer

from courserec.recommenders.lexical import bm25_weight_matrix

cnt_vec = CountVectorizer(stop_words="english")
counts = cnt_vec.fit_transform(docs).tocsr()  # raw term counts (BM25 needs tf, not tf-idf)
W = bm25_weight_matrix(counts, k1=1.5, b=0.75)  # the BM25 doc-term weights
bm_terms = np.array(cnt_vec.get_feature_names_out())
bm_row = {cid: i for i, cid in enumerate(course_ids)}


def recommend_bm25_from_scratch(seed_id: str, k: int = 10) -> list[tuple[str, float]]:
    """BM25-rank the catalog against a seed's term set (teaching version)."""
    q = (counts[bm_row[seed_id]] > 0).astype(np.float64)  # binary presence of seed terms
    scores = (W @ q.T).toarray().ravel()
    scores[bm_row[seed_id]] = -np.inf
    top = np.argsort(scores)[::-1][:k]
    return [(course_ids[i], float(scores[i])) for i in top]


bm_scratch = recommend_bm25_from_scratch(seed, k=10)
pd.DataFrame(
    [(i + 1, cid, courses.loc[cid, "title"], round(s, 2)) for i, (cid, s) in enumerate(bm_scratch)],
    columns=["rank", "course_id", "title", "bm25"],
)

# %% [markdown]
# ## 4. Sweep the knobs — watch the metric move
#
# This is the payoff of a *family* notebook: the preprocessing choices are
# **configs of one technique**, so we can sweep them and see each knob's effect on
# the headline metric. We score each config through the real cross-listing lens
# from notebook 00 (with a smaller bootstrap count so the notebook stays fast).

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist
from courserec.recommenders.lexical import BM25Recommender

truth = build_crosslist_truth(courses)
reference = build_reference_space(courses)

configs = [
    ("TF-IDF · unigram", TfidfRecommender(ngram_max=1, title_weight=1)),
    ("TF-IDF · +bigrams", TfidfRecommender(ngram_max=2, title_weight=1)),
    ("TF-IDF · title×3", TfidfRecommender(ngram_max=1, title_weight=3)),
    ("BM25 · unigram", BM25Recommender(ngram_max=1, title_weight=1)),
]

rows, labels, ndcgs, cis = [], [], [], []
for label, rec in configs:
    rec.fit(courses)  # warm-loads cached artifacts where present
    result = score_crosslist(rec, courses, truth, reference, n_boot=200)
    rows.append(
        {
            "config": label,
            "NDCG@10": round(result.metrics["ndcg@10"], 4),
            "Recall@10": round(result.metrics["recall@10"], 4),
            "MRR": round(result.metrics["mrr"], 4),
            "same_subj@10": round(result.same_subject_at_10, 3),
            "latency_ms": round(result.query_latency_ms, 2),
        }
    )
    labels.append(label)
    ndcgs.append(result.metrics["ndcg@10"])
    cis.append(result.ndcg10_ci)

pd.DataFrame(rows).sort_values("NDCG@10", ascending=False).reset_index(drop=True)

# %% [markdown]
# ## 5. The headline numbers, with error bars
#
# Plotted with the bootstrap CI whiskers from notebook 00 — because the gaps
# between these configs are small, and we never crown a winner on a sub-CI gap.

# %%
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend (works under nbmake)
import matplotlib.pyplot as plt  # noqa: E402

from nbtools import plot_metric_ci  # noqa: E402

ax = plot_metric_ci(labels, ndcgs, cis, title="Lexical configs · cross-listing lens")
plt.tight_layout()
ax.figure  # render in the notebook

# %% [markdown]
# ## 6. Takeaways
#
# - **Lexical methods nail the cross-listing lens** — twins share text, so cosine
#   overlap is high. This validates *correctness*; it does **not** prove quality.
# - **The knobs barely move the headline**, and the moves are mostly inside the
#   CIs — bag-of-words is already near its ceiling on this near-duplicate task.
#   Bigrams add vocabulary sparsity for little gain; a heavier title weight helps
#   the short-text courses slightly.
# - **Where lexical *loses*** is invisible to this lens: synonymy and paraphrase
#   ("ML" vs "machine learning", "stats" vs "statistics"). Two courses about the
#   same topic in different words share *no* terms and score ~0. That blind spot
#   is the entire reason the **semantic-vector** notebook (03) exists — come back
#   and compare.
#
# **Source:** [`courserec/recommenders/lexical.py`](../src/courserec/recommenders/lexical.py)
# · **Next:** [02 · Latent topics](02_topics.py) (LSA/NMF/LDA — compress the
# vocabulary into themes).
