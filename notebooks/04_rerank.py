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
# # 04 · Retrieve → rerank → diversify
#
# > Assumes [03](03_embeddings.py) (SBERT). Needs the `semantic` extra.
#
# Notebook 03's SBERT is a **bi-encoder**: the query and each course are embedded
# *independently*, so one index search scores all 11k courses cheaply — but the
# query and a course never "see" each other while encoding. A **cross-encoder**
# does the opposite: it feeds `(query, candidate)` through the model **together**
# and reads out a relevance score. Far more accurate, far too slow for 11k courses
# per query. The fix is two stages:
#
# 1. **Retrieve** ~50 candidates with the fast bi-encoder (high recall, cheap).
# 2. **Rerank** just those 50 with the cross-encoder (high precision, small set).
#
# Then **MMR** trades a little relevance for diversity so the top isn't three
# near-duplicates. We build each stage and watch the order change.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 50)
courses = load_processed()
seed = "COMPSCI 189"
seed_text = str(courses.loc[seed, "text"])
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. Stage one — retrieve 50 candidates (bi-encoder)
#
# Reuse the fitted SBERT rung from notebook 03 to fetch a high-recall candidate set.

# %%
from courserec.recommenders.embeddings import SbertRecommender

base = SbertRecommender(model_name="all-MiniLM-L6-v2")
base.fit(courses)  # warm
retrieved = [r.course_id for r in base.recommend_similar(seed, k=50)]
print(f"retrieved {len(retrieved)} candidates; bi-encoder top-5:", retrieved[:5])

# %% [markdown]
# ## 2. Stage two — rerank with the cross-encoder
#
# **The transformation.** Score each `(seed text, candidate text)` pair *jointly*.
# The model attends across both texts at once, so it judges relevance more sharply
# than two independent embeddings can. Watch the order change versus stage one.

# %%
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[seed_text, str(courses.loc[c, "text"])] for c in retrieved]
rel = np.asarray(ce.predict(pairs, show_progress_bar=False), dtype=np.float64)

reranked = [retrieved[i] for i in np.argsort(rel)[::-1]]
cmp = pd.DataFrame({
    "bi_encoder_rank": [retrieved[i] for i in range(8)],
    "cross_encoder_rank": reranked[:8],
})
cmp

# %% [markdown]
# ## 3. Stage three — MMR for diversity, from scratch
#
# Pure relevance order clusters near-duplicates (the cross-listed twins) at the
# top. **Maximal Marginal Relevance** picks the next item greedily by balancing
# relevance against *novelty* versus what's already chosen:
#
# `MMR(c) = λ·rel(c) − (1−λ)·max_{s∈selected} sim(c, s)`
#
# `rel` is the (min-max normalized) cross-encoder score; `sim` is cosine between
# candidates in the **bi-encoder** space. `λ=1` is pure relevance; lowering `λ`
# pushes diversity up.

# %%
# candidate vectors in the bi-encoder space (for the sim term)
cand_vecs = base._embeddings[[base._row[c] for c in retrieved]]
rel_norm = (rel - rel.min()) / (rel.max() - rel.min())  # → [0, 1]


def mmr(lam, k=8):
    """Greedy MMR selection of k candidates at trade-off lam."""
    sims = cand_vecs @ cand_vecs.T
    remaining, selected = list(range(len(retrieved))), []
    while remaining and len(selected) < k:
        max_sim = sims[np.ix_(remaining, selected)].max(axis=1) if selected else np.zeros(len(remaining))
        score = lam * rel_norm[remaining] - (1 - lam) * max_sim
        pick = remaining.pop(int(np.argmax(score)))
        selected.append(pick)
    return [retrieved[i] for i in selected]


pd.DataFrame({
    "λ=1.0 (pure relevance)": mmr(1.0),
    "λ=0.5 (diversified)": mmr(0.5),
})

# %% [markdown]
# ## 4. Cross-check against the library `RerankRecommender`

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.rerank import RerankRecommender

rer = RerankRecommender(base=base, retrieve_n=50, mmr_lambda=0.5)
rer.fit(courses)
lib = rer.recommend_similar(seed, k=8)
print(f"top-8 overlap (scratch MMR vs library): {top_k_overlap(mmr(0.5), [r.course_id for r in lib], 8):.0%}")
recs_to_frame(lib, courses)

# %% [markdown]
# ## 5. Evaluate live (on a sample of seeds)
#
# The cross-encoder runs ~50 forward passes **per seed**, so scoring all ~1,072
# cross-listing seeds is minutes of compute. For the notebook we evaluate on a
# reproducible **sample**; the full board is in `run_eval.py` / notebook 09.

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist

truth = build_crosslist_truth(courses)
rng = np.random.default_rng(42)
sample_seeds = rng.choice(sorted(truth), size=60, replace=False)
sample_truth = {s: truth[s] for s in sample_seeds}
reference = build_reference_space(courses)

for label, rec in [("SBERT base", base), ("+ rerank + MMR(0.5)", rer)]:
    r = score_crosslist(rec, courses, sample_truth, reference, n_boot=200)
    print(f"{label:22s}  NDCG@10={r.metrics['ndcg@10']:.4f}  diversity={r.diversity:.4f}  (n={r.n_queries} seeds)")

# %% [markdown]
# ## 6. Takeaways
#
# - **The cross-encoder reorders the top** (§2): judging the pair jointly is sharper
#   than two independent embeddings — but only affordable over a small retrieved set.
# - **MMR is a dial, not a model** (§3): at `λ=0.5` near-duplicates get pushed down
#   and the intra-list **diversity** metric rises, at a small relevance cost.
# - On the *cross-listing* lens the rerank barely moves NDCG@10 — the twins are
#   already at the very top from SBERT, so there's little left to sharpen. Reranking
#   earns its keep on harder, lower-precision retrieval (free-text), and MMR earns
#   its keep whenever you care about a *varied* list, not just a relevant one.
#
# **Source:** [`courserec/recommenders/rerank.py`](../src/courserec/recommenders/rerank.py)
# · **ADR:** [0005](../docs/adr/0005-rerank-mmr.md)
# · **Next:** [05 · Metadata fusion](05_metadata.py).
