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
# # 03 · Semantic vectors — SBERT embeddings
#
# > Assumes [00](00_data_and_eval.py) and [01](01_lexical.py). Needs the
# > `semantic` extra (`pip install -e ".[notebooks,semantic]"`).
#
# This is the rung that fixes lexical's blind spot. TF-IDF scored "ML" and
# "machine learning" at **zero** similarity — no shared tokens. A **sentence
# embedding** model (SBERT) maps text to a dense vector where *meaning*, not
# vocabulary, decides closeness, so paraphrases land near each other. You can't
# reimplement a transformer in a notebook, so here the "from scratch" part is the
# **retrieval machinery** around the model: encode → cosine → nearest-neighbor
# search. We start by *showing* the synonym problem disappear.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 55)
courses = load_processed()
seed = "COMPSCI 189"
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. The synonym test lexical fails
#
# Encode two phrasings of the same idea and compare them in **both** spaces.
# TF-IDF sees no shared word → 0. SBERT sees the same meaning → high cosine.

# %%
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

model = SentenceTransformer("all-MiniLM-L6-v2")  # baked into the image / cached locally
a, b = "machine learning", "ML"

# lexical similarity
tf = TfidfVectorizer().fit([a, b])
lex = (tf.transform([a]) @ tf.transform([b]).T).toarray()[0, 0]

# semantic similarity (encode → L2-normalize → dot = cosine)
va, vb = model.encode([a, b], normalize_embeddings=True)
print(f"'{a}'  vs  '{b}'")
print(f"   TF-IDF cosine : {lex:.3f}   ← no shared token, so zero")
print(f"   SBERT  cosine : {float(va @ vb):.3f}   ← same meaning, so high")

# %% [markdown]
# ## 2. The catalog, already encoded
#
# Encoding ~11k courses takes a while, so the SBERT rung **caches** its vectors to
# `artifacts/`. We load that matrix (the same one the production recommender and
# the Map view use) instead of re-encoding — `load_sbert_embeddings` fits an
# `SbertRecommender`, which reloads the artifact warm.

# %%
from courserec.cluster import load_sbert_embeddings

E, ids = load_sbert_embeddings(courses)  # (n_courses × 384) L2-normalized
row_of = {c: i for i, c in enumerate(ids)}
print(f"embedding matrix: {E.shape[0]:,} courses × {E.shape[1]} dims (unit-norm rows)")

# %% [markdown]
# ## 3. Nearest-neighbor search, by hand
#
# With unit-norm rows, cosine similarity to the seed is one matrix-vector product.
# Rank, drop the seed, take the top-k — the same move as every prior rung, now in
# *learned* space.

# %%
sims = E @ E[row_of[seed]]
sims[row_of[seed]] = -np.inf
top = np.argsort(sims)[::-1][:10]
pd.DataFrame(
    [(i + 1, ids[j], courses.loc[ids[j], "title"], round(float(sims[j]), 3)) for i, j in enumerate(top)],
    columns=["rank", "course_id", "title", "cosine"],
)

# %% [markdown]
# ## 4. The same search via FAISS (the ANN index)
#
# At 11k courses a brute-force dot product is instant, but production retrieval
# scales with an **approximate nearest-neighbor** index. The library builds a FAISS
# inner-product index over the same vectors; for unit-norm vectors inner product =
# cosine, so on a flat index it returns the *identical* neighbors — just via an
# index built to scale to millions.

# %%
import faiss

index = faiss.IndexFlatIP(E.shape[1])
index.add(E.astype(np.float32))
_, idx = index.search(E[row_of[seed]].reshape(1, -1).astype(np.float32), 11)
faiss_ids = [ids[j] for j in idx[0] if ids[j] != seed][:10]
print("FAISS top-5:", faiss_ids[:5])

# %% [markdown]
# ## 5. Cross-check against the library `SbertRecommender`

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.embeddings import SbertRecommender

sbert = SbertRecommender(model_name="all-MiniLM-L6-v2")
sbert.fit(courses)  # warm-loads the cached embeddings + FAISS index
lib = sbert.recommend_similar(seed, k=10)
print(f"top-10 overlap (by-hand vs library): {top_k_overlap([ids[j] for j in top], [r.course_id for r in lib], 10):.0%}")
recs_to_frame(lib, courses)

# %% [markdown]
# ## 6. Free-text search — the mode lexical can't really do
#
# Because queries and courses share one space, a natural-language query encodes to
# a vector and retrieves by meaning. This is what the judged-query lens measures.

# %%
for q in ["practical deep learning", "ethics of technology"]:
    hits = sbert.recommend_by_text(q, k=3)
    print(f"\n{q!r}:")
    for r in hits:
        print(f"   {r.course_id:12s} {courses.loc[r.course_id, 'title']}")

# %% [markdown]
# ## 7. Evaluate live

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist

truth = build_crosslist_truth(courses)
reference = build_reference_space(courses)
res = score_crosslist(sbert, courses, truth, reference, n_boot=200)
print("SBERT MiniLM — cross-listing lens:")
print(f"  NDCG@10   = {res.metrics['ndcg@10']:.4f}  (95% CI {res.ndcg10_ci[0]:.3f}–{res.ndcg10_ci[1]:.3f})")
print(f"  Recall@10 = {res.metrics['recall@10']:.4f}")

# %% [markdown]
# ## 8. Takeaways
#
# - **Semantic vectors top the leaderboard** and, unlike lexical, actually do
#   free-text search — meaning beats vocabulary (§1, §6).
# - The retrieval *mechanism* is the same one-mat-vec cosine as every prior rung;
#   what changed is the **space**: learned, dense, 384-D instead of sparse term
#   counts. FAISS is how that search scales past a brute-force dot product (§4).
# - Cost: you trade interpretability (a 384-D vector has no readable axes, unlike
#   notebook 02's topics) and a model download/encode for the quality.
#
# **Source:** [`courserec/recommenders/embeddings.py`](../src/courserec/recommenders/embeddings.py)
# · **ADR:** [0004](../docs/adr/0004-semantic-vectors.md)
# · **Next:** [04 · Retrieve → rerank](04_rerank.py) — make the top of the list
# sharper and more diverse.
