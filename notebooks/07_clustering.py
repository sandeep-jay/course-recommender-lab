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
# # 07 · Clustering + the 2-D map
#
# > Assumes [03](03_embeddings.py) (SBERT vectors). Needs the `semantic` extra.
#
# Every other notebook *ranks*. This one **doesn't** — it's a **diagnostic** that
# asks a shape question: do the SBERT embeddings organize into coherent groups, and
# do those groups line up with the catalog's own structure (subjects)? If text
# embeddings recover subjects *with no metadata*, that's strong evidence the space
# is meaningful. The two outputs are a coherence table and a map of the whole
# catalog you can actually look at.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed
from courserec.cluster import load_sbert_embeddings

pd.set_option("display.max_colwidth", 55)
courses = load_processed()
E, ids = load_sbert_embeddings(courses)  # warm: the cached SBERT matrix
subjects = courses.loc[ids, "subject"].to_numpy()
print(f"{E.shape[0]:,} embeddings × {E.shape[1]} dims · {len(set(subjects))} subjects")

# %% [markdown]
# ## 1. Partition the embeddings (KMeans)
#
# **The transformation.** Group the 11k unit vectors into `k=100` clusters by
# minimizing within-cluster variance. On unit-norm vectors, squared-Euclidean is
# monotone in cosine, so this is effectively spherical k-means — clusters of
# courses pointing the same semantic direction.

# %%
from courserec.cluster import cluster_embeddings

labels, config = cluster_embeddings(E, algorithm="kmeans", n_clusters=100)
sizes = pd.Series(labels).value_counts()
print(f"{config} → {len(sizes)} clusters, sizes from {sizes.min()} to {sizes.max()}")

# %% [markdown]
# ## 2. Is a cluster coherent? Look inside one
#
# Pick a mid-sized cluster and read its members. Coherent embeddings → a readable
# theme without anyone labeling it.

# %%
target = sizes.index[5]
members = [ids[i] for i in np.where(labels == target)[0]][:8]
pd.DataFrame(
    [(cid, courses.loc[cid, "subject"], courses.loc[cid, "title"]) for cid in members],
    columns=["course_id", "subject", "title"],
)

# %% [markdown]
# ## 3. Score coherence — internal and external
#
# - **Silhouette** (internal): how much tighter a point sits with its own cluster
#   than the next-nearest, in `[-1, 1]`.
# - **Subject purity** (external): the size-weighted share of each cluster's
#   dominant subject. High purity = clusters recovered *subjects from text alone*.
#
# We compare a forced-`k` method (KMeans) with a density method (**HDBSCAN**), which
# picks its own cluster count and can label sparse regions as **noise** (`-1`).

# %%
from courserec.cluster import evaluate_clustering

rows = []
for algo in ("kmeans", "hdbscan"):
    lab, cfg = cluster_embeddings(E, algorithm=algo, n_clusters=100)
    res = evaluate_clustering(E, lab, subjects, algorithm=algo, config=cfg)
    rows.append(res.summary_row())
pd.DataFrame(rows)[["algorithm", "n_clusters", "n_noise", "silhouette", "subject_purity", "largest_cluster_frac"]]

# %% [markdown]
# ## 4. The map — project 11k dims to 2
#
# **The transformation.** A non-linear projection (t-SNE here; UMAP if the `viz`
# extra is installed) lays the catalog out in 2-D so that *visual* proximity ≈
# *embedding* proximity. Colored by subject, clusters of one color confirm the
# space groups related courses. (This is the static version of the UI's Map view.)

# %%
import matplotlib
matplotlib.use("Agg")  # headless backend (works under nbmake)
import matplotlib.pyplot as plt

from courserec.cluster import project_2d

coords, method = project_2d(E, method="auto")  # t-SNE fallback when umap-learn is absent
top_subjects = pd.Series(subjects).value_counts().head(12).index.tolist()

fig, ax = plt.subplots(figsize=(11, 8))
other = ~np.isin(subjects, top_subjects)
ax.scatter(coords[other, 0], coords[other, 1], s=3, c="lightgray", alpha=0.4)
cmap = plt.get_cmap("tab20")
for i, subj in enumerate(top_subjects):
    m = subjects == subj
    ax.scatter(coords[m, 0], coords[m, 1], s=6, color=cmap(i % 20), label=subj)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Catalog embedding map ({method}) — top 12 subjects")
ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.9)
fig.tight_layout()
fig

# %% [markdown]
# ## 5. Takeaways
#
# - **Text recovers structure:** subject purity well above chance means the SBERT
#   space groups courses into subject-like neighborhoods with **no metadata** —
#   independent evidence (beyond the rankers' scores) that the embeddings are
#   meaningful.
# - **HDBSCAN is the honest counterpoint** to KMeans: it can say "this region is
#   noise" instead of forcing every course into a cluster, so its noise fraction and
#   cluster count are a reality check on the forced-`k` view.
# - **Not a recommender:** there is no ranking here and it never joins the
#   leaderboard. It feeds the *coverage / diversity* story the rankers can only
#   report as scalars, and it's the diagnostic behind the UI's Map view.
#
# **Source:** [`courserec/cluster.py`](../src/courserec/cluster.py)
# · **ADR:** [0007](../docs/adr/0007-clustering-diagnostic.md)
# · **Next:** [08 · LLM enrichment](08_llm.py).
