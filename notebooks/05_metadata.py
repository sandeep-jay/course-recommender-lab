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
# # 05 · Metadata fusion — blend structure with text
#
# > Assumes [00](00_data_and_eval.py) and [01](01_lexical.py) (TF-IDF).
#
# Every rung so far compares courses on **one** signal: words. But the catalog
# also carries cheap, perfectly-clean **structure** the text ignores — each
# course's `subject`, `department`, `level`, and `units`. Two courses can read
# very differently yet both be graduate Mechanical Engineering seminars. This
# technique **fuses** a TF-IDF text vector with a one-hot metadata vector into one
# blended vector and ranks by their weighted combination.
#
# We use TF-IDF (not SBERT) as the text half on purpose: it makes the metadata's
# contribution a clean **ablation** — `metadata(...)` differs from the plain
# `tfidf(...)` baseline *only* by the fused structure, so any change is
# attributable to the metadata, not a fancier encoder.

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 55)
courses = load_processed()
seed = "MECENG 203"
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. Block one — the TF-IDF text vector
#
# Exactly notebook 01's matrix; rows are already L2-normalized.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

ids = list(courses.index)
row_of = {c: i for i, c in enumerate(ids)}
text_vec = TfidfVectorizer(stop_words="english")
T = text_vec.fit_transform(courses["text"].fillna(""))
print(f"text block:  {T.shape[0]:,} × {T.shape[1]:,} terms")

# %% [markdown]
# ## 2. Block two — one-hot the structured facets
#
# **The transformation.** Turn each course's `subject / department / level / units`
# into indicator columns (`subject=MECENG`, `level=grad`, …). `DictVectorizer`
# builds the sparse one-hot matrix; we L2-normalize it so the metadata block is the
# same scale as the text block.

# %%
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize


def facet_dict(cid):
    """One course's active facets as a {name=value: 1} indicator dict."""
    r = courses.loc[cid]
    d = {}
    for facet, col in [("subject", "subject"), ("department", "department"), ("level", "level"), ("units", "units_min")]:
        v = r[col]
        if pd.notna(v):
            d[f"{facet}={v:g}" if facet == "units" else f"{facet}={v}"] = 1
    return d


M = normalize(DictVectorizer(sparse=True).fit_transform([facet_dict(c) for c in ids]))
print(f"meta block:  {M.shape[0]:,} × {M.shape[1]:,} facet indicators")
print(f"seed's facets: {facet_dict(seed)}")

# %% [markdown]
# ## 3. Fuse — scale each block and concatenate
#
# **The fusion knob `λ = text_weight`.** Scale the text block by `λ`, the metadata
# block by `1 − λ`, and stack them side by side into one vector:
#
# `v(d) = [ λ·t(d) ‖ (1−λ)·m(d) ]`
#
# Because each block is unit-norm, the dot product of two fused vectors is
# `λ²·cos_text + (1−λ)²·cos_meta`. So `λ=1` is pure TF-IDF, `λ=0` is pure metadata,
# and values between tune how hard the structure pulls. We use `λ=0.7`.

# %%
lam = 0.7
fused = sp.hstack([T * lam, M * (1.0 - lam)]).tocsr()
print(f"fused vector: {fused.shape[1]:,} dims = {T.shape[1]:,} text + {M.shape[1]:,} meta")

# %% [markdown]
# ## 4. Rank with the fused vectors
#
# One sparse mat-vec against the seed's fused row, drop the seed, take the top-k.

# %%
scores = np.asarray((fused @ fused[row_of[seed]].T).todense()).ravel()
scores[row_of[seed]] = -np.inf
top = np.argsort(scores)[::-1][:10]
pd.DataFrame(
    [(i + 1, ids[j], courses.loc[ids[j], "subject"], courses.loc[ids[j], "title"], round(float(scores[j]), 3))
     for i, j in enumerate(top)],
    columns=["rank", "course_id", "subject", "title", "fused"],
)

# %% [markdown]
# ## 5. Cross-check against the library `MetadataRecommender`

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.metadata import MetadataRecommender

meta_rec = MetadataRecommender(text_weight=0.7)
meta_rec.fit(courses)  # warm-loads the cached artifact
lib = meta_rec.recommend_similar(seed, k=10)
print(f"top-10 overlap (scratch vs library): {top_k_overlap([ids[j] for j in top], [r.course_id for r in lib], 10):.0%}")
recs_to_frame(lib, courses)

# %% [markdown]
# ## 6. Evaluate live — and an honest result
#
# We score the fusion against the **plain TF-IDF baseline** on the cross-listing
# lens, so the metadata's effect is isolated.

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist
from courserec.recommenders.lexical import TfidfRecommender

truth = build_crosslist_truth(courses)
reference = build_reference_space(courses)

tfidf = TfidfRecommender(ngram_max=1, title_weight=1)
tfidf.fit(courses)
for label, rec in [("TF-IDF (text only)", tfidf), ("Metadata fusion (λ=0.7)", meta_rec)]:
    r = score_crosslist(rec, courses, truth, reference, n_boot=200)
    print(f"{label:26s}  NDCG@10={r.metrics['ndcg@10']:.4f}  (CI {r.ndcg10_ci[0]:.3f}–{r.ndcg10_ci[1]:.3f})")

# %% [markdown]
# ## 7. Takeaways
#
# - **Metadata fusion *hurts* the cross-listing target** — and that is the honest,
#   interesting result. Cross-listed twins already share near-identical *text*;
#   mixing in `subject`/`department` pulls in same-department courses that are
#   **not** the twin, pushing the twin down. The metadata block adds noise to a
#   signal text already solves.
# - **Where it would help is a different goal:** "show me courses *like* this one in
#   spirit and department" — coherence the cross-listing lens is the wrong ruler
#   for. And it rescues **sparse-text courses**: a one-line description has a nearly
#   empty text block but a fully-populated metadata block, so fusion still ranks it.
# - The clean-ablation design (TF-IDF text half) is what lets us *attribute* the
#   drop to the structure rather than guess.
#
# **Source:** [`courserec/recommenders/metadata.py`](../src/courserec/recommenders/metadata.py)
# · **ADR:** [0008](../docs/adr/0008-metadata-fusion.md)
# · **Next:** [06 · Course graph](06_graph.py).
