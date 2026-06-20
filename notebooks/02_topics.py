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
# # 02 · Latent topics — LSA, NMF, LDA
#
# > Assumes [00](00_data_and_eval.py) (data + eval) and [01](01_lexical.py) (TF-IDF).
#
# Lexical methods compare courses in **raw term space** — tens of thousands of
# sparse dimensions, one per word, with no notion that "statistics" and
# "statistical" are related. **Topic models** first *reduce* that space to a few
# hundred dense **topics** (combinations of co-occurring words), then compare
# courses there. Two payoffs: the topic vector is small and dense (synonyms that
# co-occur collapse onto a shared topic), and the topic→term tables are
# **readable** — you can see what each axis means.
#
# We build the workhorse — **LSA** (truncated SVD of the TF-IDF matrix) — from
# scratch, then contrast NMF and LDA, then evaluate live.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 55)
RANDOM_SEED = 42

courses = load_processed()
seed = "COMPSCI 189"
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. Start where lexical left off: the TF-IDF matrix
#
# The input to LSA and NMF is exactly the TF-IDF document-term matrix from
# notebook 01 — `n_courses × n_terms`, sparse. Topic modeling is what we do *to*
# that matrix.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

docs = courses["text"].fillna("").tolist()
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)  # the same sparse term matrix as notebook 01
ids = list(courses.index)
row_of = {c: i for i, c in enumerate(ids)}
terms = np.array(vec.get_feature_names_out())
print(f"TF-IDF: {X.shape[0]:,} courses × {X.shape[1]:,} terms")

# %% [markdown]
# ## 2. LSA = truncated SVD of that matrix
#
# **The transformation.** Factor `X ≈ U Σ Vᵀ`, keeping only the top `k` singular
# directions. Each retained direction is a **topic** — a weighted blend of terms.
# A course's **topic vector** is its row of `U·Σ` (`TruncatedSVD.transform`). We
# go from ~30k sparse term dims down to `k=200` dense topic dims.

# %%
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=200, random_state=RANDOM_SEED)
doc_topics = svd.fit_transform(X)  # (n_courses × 200) dense
print(f"reduced: {X.shape[1]:,} term dims → {doc_topics.shape[1]} topic dims")
print(f"variance explained by 200 topics: {svd.explained_variance_ratio_.sum():.1%}")

# %% [markdown]
# ## 3. What does a topic mean? (the interpretability payoff)
#
# Each topic is a row of `svd.components_` (`n_topics × n_terms`) — the term
# loadings that define that axis. Reading the highest-loading terms tells you the
# theme. This is what topic models give you that lexical never could.

# %%
for t in [0, 5, 12]:
    top_terms = terms[np.argsort(svd.components_[t])[::-1][:8]]
    print(f"topic {t:3d}: {', '.join(top_terms)}")

# %% [markdown]
# ## 4. Rank in topic space
#
# Same recipe as lexical, one space lower: L2-normalize the topic vectors so a dot
# product is cosine, then rank the catalog against the seed's topic vector and drop
# the seed. (Topic similarities can be negative — LSA's axes are signed — so unlike
# lexical we don't filter to positive scores, we just rank.)

# %%
def l2(m):
    """Row-normalize so a dot product equals cosine (in-place-safe copy)."""
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


DT = l2(doc_topics.astype(np.float64))
sims = DT @ DT[row_of[seed]]
sims[row_of[seed]] = -np.inf
top = np.argsort(sims)[::-1][:10]
pd.DataFrame(
    [(i + 1, ids[j], courses.loc[ids[j], "title"], round(float(sims[j]), 3)) for i, j in enumerate(top)],
    columns=["rank", "course_id", "title", "cosine(topic)"],
)

# %% [markdown]
# ## 5. Cross-check against the library `LSARecommender`

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.topics import LSARecommender

lsa = LSARecommender(n_topics=200)
lsa.fit(courses)  # warm-loads the cached artifact
lib = lsa.recommend_similar(seed, k=10)
print(f"top-10 overlap (scratch vs library): {top_k_overlap([ids[j] for j in top], [r.course_id for r in lib], 10):.0%}")
recs_to_frame(lib, courses)

# %% [markdown]
# ## 6. NMF and LDA — two other ways to find topics
#
# - **NMF** factors `X ≈ W H` with everything **non-negative**, so topics read as
#   additive "parts" (often cleaner themes than LSA's signed axes).
# - **LDA** is a *probabilistic* model over raw word **counts**: each course is a
#   mixture of topics, each topic a distribution over words.
#
# Same interface, so we just read their topic-term tables to compare flavor. (LDA
# is the slowest to fit, so we use a small `k` here purely to illustrate.)

# %%
from courserec.recommenders.topics import NMFRecommender

nmf = NMFRecommender(n_topics=20)
nmf.fit(courses)  # warm-loads cache
print("NMF additive topics (first 4):")
for t in range(4):
    print(f"  topic {t}: {', '.join(nmf.topic_terms(t, n=7))}")

# %% [markdown]
# ## 7. Evaluate live — does reducing to topics help on cross-listings?

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist

truth = build_crosslist_truth(courses)
reference = build_reference_space(courses)
res = score_crosslist(lsa, courses, truth, reference, n_boot=200)
print(f"LSA (200 topics) — cross-listing lens:")
print(f"  NDCG@10     = {res.metrics['ndcg@10']:.4f}  (95% CI {res.ndcg10_ci[0]:.3f}–{res.ndcg10_ci[1]:.3f})")
print(f"  Recall@10   = {res.metrics['recall@10']:.4f}")
print(f"  MRR         = {res.metrics['mrr']:.4f}")

# %% [markdown]
# ## 8. Takeaways
#
# - **Topic models roughly *match*, not beat, lexical on the cross-listing lens.**
#   Twins already share near-identical text, so compressing to 200 topics can only
#   *blur* a signal lexical already nails — the CI overlaps the TF-IDF baseline.
# - **The real payoff is elsewhere:** (1) **interpretability** — you can read what
#   each topic axis means (§3), which raw term space can't give you; (2) **free-text
#   robustness** — synonyms collapse onto shared topics, which the judged-query lens
#   (notebook 09) rewards more than the near-duplicate cross-listing lens does.
# - **LSA vs NMF vs LDA** trade speed for interpretability: LSA is fast and signed,
#   NMF gives cleaner additive parts, LDA is the principled-but-slow count model.
#
# **Source:** [`courserec/recommenders/topics.py`](../src/courserec/recommenders/topics.py)
# · **Next:** [03 · Semantic vectors](03_embeddings.py) — dense *learned* meaning,
# the rung that finally fixes lexical's synonym blindness.
