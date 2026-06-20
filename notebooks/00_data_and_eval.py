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
# # 00 · The dataset and the evaluation harness
#
# **This is the foundation notebook — every technique notebook (01–08) assumes
# it.** It teaches the two things that are *shared* across all techniques, so the
# per-technique notebooks never have to re-explain them:
#
# 1. **The data** — how the UC Berkeley catalog is cleaned into a model-ready
#    frame (the `"-"` null token, the synthesized `course_id`, the sparse-text
#    fallback), and the one column you must **never** feed a model.
# 2. **The evaluation harness** — how we measure "good" with *no* clicks or
#    ratings: the three lenses, the ranking metrics, and why every headline number
#    carries a bootstrap confidence interval.
#
# > **Reuse, not reimplement.** Unlike the technique notebooks, this one *uses*
# > the library (`courserec.data`, `courserec.eval`) directly — data loading and
# > the metric definitions are plumbing the whole project shares, not a technique
# > being taught. The metrics are *defined* here; the first **live** numbers appear
# > in notebook 01, computed on a real model's ranking (every technique notebook
# > runs its own eval).

# %%
import pandas as pd

from courserec.data import load_processed

pd.set_option("display.max_colwidth", 70)
RANDOM_SEED = 42  # the project-wide seed; every stochastic step pins it

courses = load_processed()
print(f"{len(courses):,} courses × {courses.shape[1]} columns")
courses.head(3)

# %% [markdown]
# ## 1. What one course looks like
#
# The frame is **indexed by `course_id`** — synthesized as `f"{Subject} {Course
# Number}"` (e.g. `COMPSCI 189`), because the raw catalog has no single stable key.
# The `text` column is the model input for every text-based technique: the title,
# then the description appended. Let's look at one row in full.

# %%
seed = "COMPSCI 189"
row = courses.loc[seed]
for col in ["subject", "department", "title", "level", "units_min", "cross_listed"]:
    print(f"{col:14s}: {row[col]}")
print(f"\ntext (model input):\n  {row['text'][:240]}…")

# %% [markdown]
# ### The `"-"` null token (load-bearing rule #1)
#
# The catalog uses the **string `"-"`** as its null value. If you treat it as data
# you get a phantom "course about `-`". The loader replaces it with real `NA` on
# read, so by the time we see the frame, missingness is honest `NaN` — and we can
# count it.

# %%
print("Missing values per column (after the '-' → NA fix):")
print(courses.isna().sum().to_string())
n_sparse = (courses["description"].isna()).sum()
print(f"\n{n_sparse:,} courses have no description — text-based techniques must")
print("fall back to the title for these, never crash. (load-bearing rule #3)")

# %% [markdown]
# ## 2. The ground-truth column you must never use as a feature
#
# **Load-bearing rule #2 (leakage):** `cross_listed` — the courses the registrar
# declares are *the same class under two department codes* — is our **evaluation
# ground truth**. A cross-listed twin is the closest thing to a labeled "these two
# are equivalent" pair the catalog gives us. So **no technique may read
# `cross_listed` as an input feature** (the graph model is the one exception, and
# it must hold out a split of these edges — see notebook 06).
#
# The harness turns this column into a `seed → {twins}` mapping.

# %%
from courserec.eval import build_crosslist_truth

truth = build_crosslist_truth(courses)
print(f"{len(truth):,} courses have at least one in-catalog cross-listed twin")
print(f"  → {100 * len(truth) / len(courses):.1f}% of the catalog carries a label\n")

example = next(c for c in truth if c.startswith("COMPSCI"))
print(f"example: {example}  ({courses.loc[example, 'title']})")
for twin in truth[example]:
    print(f"   twin: {twin}  ({courses.loc[twin, 'title']})")

# %% [markdown]
# Notice the twins share **near-identical text** — that is exactly why this lens
# *validates correctness more than quality*: even a dumb keyword matcher will rank
# a twin near the top. We keep it as the primary lens because it is the only fully
# automatic relevance signal, but we never trust it alone.

# %% [markdown]
# ## 3. The three lenses — why no single one is enough
#
# | Lens | What it measures | Trust level |
# |---|---|---|
# | **Cross-listing pairs** | does a course rank its declared twin near the top? | primary, automatic — but near-duplicate text makes it easy |
# | **Same-subject coherence** | fraction of the top-k in the seed's subject | a *sanity floor* only — a subject-only model maxes it while being useless |
# | **Judged text queries** | for free-text search: ~30 hand-labeled queries → relevant courses | the **only** way to score `recommend_by_text` |
#
# The metrics below are computed identically across all three; only the source of
# the *ranking* and the *relevant set* changes.

# %% [markdown]
# ## 4. The ranking metrics
#
# Every metric takes a **ranked list of `course_id`s** and a **set of relevant
# ids** and reduces it to a number in `[0, 1]`. They are defined in
# [`courserec/eval.py`](../src/courserec/eval.py); here is what each rewards:
#
# | Metric | Question it answers |
# |---|---|
# | **Recall@k** | of all the relevant courses, what fraction made the top-k? |
# | **Precision@k** | of the top-k we returned, what fraction were relevant? |
# | **MRR** | how high is the *first* relevant hit? (`1/rank` of that hit) |
# | **MAP** | precision averaged at each relevant hit — rewards packing hits early |
# | **NDCG@k** | like recall, but **discounted by rank** (`1/log2(rank+1)`) and normalized to `[0,1]` — a hit at rank 1 is worth more than one at rank 10 |
#
# **NDCG@10 is the headline metric**: it captures not just *whether* the right
# courses appear, but whether they appear *high*. We don't compute these on a
# made-up list — the first real numbers come in notebook 01, where a fitted TF-IDF
# model produces an actual ranking to score.

# %% [markdown]
# ## 5. Why every headline number carries error bars
#
# Only ~10% of courses carry a cross-listing label, so the evaluation set is
# **small** — a few hundred to a thousand queries. With a small sample, a 0.01 gap
# in mean NDCG@10 between two techniques can be noise. So the harness reports a
# **percentile bootstrap confidence interval** (`bootstrap_ci`): resample the
# per-query scores with replacement many times and read off the 2.5th / 97.5th
# percentiles of the mean. Two techniques whose CIs overlap are a tie — **never
# crown a winner on a sub-CI gap.** Each technique notebook prints this CI beside
# its NDCG@10.

# %% [markdown]
# ## 6. Putting it together
#
# The harness wires all of this into one call per technique. `score_crosslist`
# takes a *fitted* recommender and returns an `EvalResult` with every metric, the
# NDCG@10 CI, the list-quality numbers (coverage, diversity, novelty), and timing
# — one leaderboard row. We don't fit a model here (that's notebook 01 onward);
# we just show the shared pieces it needs.

# %%
from courserec.eval import build_reference_space

# The diversity reference space — a *technique-agnostic* TF-IDF space, so no model
# is flattered by being scored in its own geometry.
reference = build_reference_space(courses)
ref_matrix, _ = reference
print(f"reference space: {ref_matrix.shape[0]:,} courses × {ref_matrix.shape[1]:,} terms")
print("\nThe inputs every technique's eval shares:")
print(f"  • catalog        : {len(courses):,} courses")
print(f"  • crosslist truth: {len(truth):,} labeled seeds")
print(f"  • reference space: for intra-list diversity")
print("\nNext: notebook 01 fits the first real techniques (TF-IDF, BM25) and runs")
print("them through exactly this harness.")

# %% [markdown]
# ---
# ### Takeaways
# - The catalog is cleaned to one row per `course_id`; `"-"` is null, sparse text
#   falls back to the title.
# - `cross_listed` is **ground truth, never a feature** (the leakage rule).
# - Three lenses, never one: cross-listing (primary), same-subject (floor only),
#   judged text (the only free-text signal).
# - Headline metric = **NDCG@10 with a bootstrap CI**.
#
# **Source:** [`courserec/data.py`](../src/courserec/data.py),
# [`courserec/eval.py`](../src/courserec/eval.py) ·
# **ADRs:** [0001](../docs/adr/0001-duplicate-course-ids.md),
# [0002](../docs/adr/0002-eval-harness-design.md),
# [0003](../docs/adr/0003-judged-query-lens.md)
