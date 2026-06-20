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
# # 06 · Course graph — personalized PageRank
#
# > Assumes [00](00_data_and_eval.py) (esp. the **leakage rule**).
#
# Every other technique compares *content*. This one compares **position in a
# graph**: build a network of courses and rank by proximity to the seed, the way
# PageRank ranks web pages. It is also the **one technique allowed to read
# `cross_listed`** as an input — so it must be evaluated *only on cross-listing
# edges it never saw* (a held-out split), or it would be grading its own answer key.
#
# We build the graph and the random-walk-with-restart from scratch, then show the
# held-out evaluation that keeps it honest.

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp

from courserec.data import load_processed
from courserec.eval import crosslist_edges

pd.set_option("display.max_colwidth", 55)
courses = load_processed()
ids = list(courses.index)
row_of = {c: i for i, c in enumerate(ids)}
n = len(ids)
seed = "STAT C241A"
print(f"{len(courses):,} courses · seed = {seed} ({courses.loc[seed, 'title']})")

# %% [markdown]
# ## 1. Build the graph
#
# **Nodes:** every course, plus lightweight **auxiliary nodes** — one per subject
# and one per department. A course links to its subject node and department node,
# so two courses in the same subject are two hops apart *through* that node (a star,
# not a dense clique). **Edges:**
#
# - course ↔ cross-listed twin — the strong, sparse signal (weight `1.0`)
# - course ↔ its subject / department aux node — weak, dense glue (weight `0.3`)
#
# `crosslist_edges` resolves the raw `cross_listed` column into clean `{a, b}` edges
# (plumbing); the graph assembly is the lesson.

# %%
rows, cols, data = [], [], []


def add(i, j, w):
    """Add an undirected weighted edge (both directions)."""
    rows.extend((i, j)); cols.extend((j, i)); data.extend((w, w))


for a, b in (tuple(e) for e in crosslist_edges(courses)):
    add(row_of[a], row_of[b], 1.0)  # cross-listing edges

next_id = n  # aux nodes are appended after the course block
group_node = {}
for col in ("subject", "department"):
    for cid, val in courses[col].items():
        if not isinstance(val, str) or not val:
            continue
        key = (col, val)
        if key not in group_node:
            group_node[key] = next_id; next_id += 1
        add(row_of[cid], group_node[key], 0.3)  # metadata glue

n_nodes = next_id
A = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float64)
A.sum_duplicates()
print(f"graph: {n:,} course nodes + {len(group_node):,} aux nodes = {n_nodes:,} nodes, {A.nnz:,} directed edges")

# %% [markdown]
# ## 2. Random walk with restart (personalized PageRank)
#
# **The transformation.** Imagine a walker starting at the seed. Each step it moves
# to a random neighbor, but with probability `c` (the *restart*) it teleports back
# to the seed. The stationary visit distribution `r` scores every node by how
# "close to the seed" it is. We solve the fixed point
#
# `r = (1−c)·A·(r ⊙ 1/deg) + c·e_seed`
#
# by **power iteration** — a handful of sparse mat-vecs. `A·(r ⊙ 1/deg)` applies the
# column-stochastic transition without ever building a dense matrix.

# %%
deg = np.asarray(A.sum(axis=1)).ravel()
inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
c = 0.15  # restart probability
e = np.zeros(n_nodes); e[row_of[seed]] = 1.0

r = e.copy()
for step in range(60):
    nxt = (1.0 - c) * (A @ (r * inv_deg)) + c * e
    if np.abs(nxt - r).sum() < 1e-6:
        r = nxt; print(f"converged after {step + 1} iterations"); break
    r = nxt

# %% [markdown]
# ## 3. Read off the recommendations
#
# Keep only the **course** nodes (drop the aux nodes and the seed), rank by visit
# score. The seed's twin and its graph-neighbors rise to the top.

# %%
course_scores = r[:n].copy()
course_scores[row_of[seed]] = -np.inf
top = np.argsort(course_scores)[::-1][:10]
pd.DataFrame(
    [(i + 1, ids[j], courses.loc[ids[j], "subject"], courses.loc[ids[j], "title"], round(float(course_scores[j]), 5))
     for i, j in enumerate(top)],
    columns=["rank", "course_id", "subject", "title", "ppr"],
)

# %% [markdown]
# ## 4. Cross-check against the library `GraphRecommender`
#
# (Built on the *full* graph here, to match our from-scratch walk above.)

# %%
from nbtools import recs_to_frame, top_k_overlap

from courserec.recommenders.graph import GraphRecommender

graph = GraphRecommender(use_metadata=True, w_xlist=1.0, w_meta=0.3, restart=0.15)
graph.fit(courses)  # full graph (no held-out) — for the mechanism cross-check only
lib = graph.recommend_similar(seed, k=10)
print(f"top-10 overlap (scratch vs library): {top_k_overlap([ids[j] for j in top], [r.course_id for r in lib], 10):.0%}")
recs_to_frame(lib, courses)

# %% [markdown]
# ## 5. The leakage-free evaluation
#
# Because the graph *reads* cross-listings, scoring it on all of them is cheating.
# `split_crosslist_edges` holds out 30% of the edges; we **remove those from the
# graph** and ask the model to recover them. This is the only fair number — and it
# is reported on its **own** leaderboard, never compared to the content rungs'.

# %%
from courserec.eval import (
    build_crosslist_truth,
    build_reference_space,
    score_crosslist,
    split_crosslist_edges,
)

split = split_crosslist_edges(build_crosslist_truth(courses))
print(f"held out {len(split.held_out_edges):,} edges; the graph never sees them")

held = GraphRecommender(use_metadata=True, held_out_edges=split.held_out_edges)
held.fit(courses)  # graph built WITHOUT the held-out edges
reference = build_reference_space(courses)
res = score_crosslist(held, courses, split.test_truth, reference, n_boot=200)
print(f"\nGraph (held-out edges) — recover-the-removed-twin lens:")
print(f"  NDCG@10   = {res.metrics['ndcg@10']:.4f}  (95% CI {res.ndcg10_ci[0]:.3f}–{res.ndcg10_ci[1]:.3f})")
print(f"  Recall@10 = {res.metrics['recall@10']:.4f}  on {res.n_queries:,} held-out seeds")

# %% [markdown]
# ## 6. Takeaways
#
# - **Recommendation as graph proximity:** no text at all — a course is "similar"
#   if a random walker keeps landing on it. The metadata glue lets the walk reach
#   same-subject neighbors even when the only cross-listing edge was held out.
# - **Wins** when a held-out twin is reachable through *remaining* structure (a
#   three-way cross-listing, or a shared subject). **Loses** on isolated pairs: if
#   the only link between two twins is the removed edge and they share no
#   subject/department, no walk can recover it — an honest ceiling.
# - **Item-to-item only:** the graph has no text encoder, so `recommend_by_text`
#   raises `NotImplementedError` — and its score lives on a *separate* board,
#   because "recover a held-out edge" is not the same task as the content rungs'.
#
# **Source:** [`courserec/recommenders/graph.py`](../src/courserec/recommenders/graph.py)
# · **ADR:** [0006](../docs/adr/0006-graph-heldout.md)
# · **Next:** [07 · Clustering + map](07_clustering.py).
