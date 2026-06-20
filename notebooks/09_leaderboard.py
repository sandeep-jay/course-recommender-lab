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
# # 09 · The leaderboard — what won, and why
#
# > Assumes you've skimmed 01–08. This notebook *synthesizes* them.
#
# The whole point of the project: every technique implements one interface and is
# scored by one harness, so they're directly comparable on one board. Each
# technique notebook ran its **own** eval live; this notebook reads the **canonical**
# board that `scripts/run_eval.py` writes (the single source of truth — we don't
# re-derive 18 rows here) and tells the story across all of them.

# %%
import pandas as pd

from courserec.config import RESULTS_DIR

pd.set_option("display.max_colwidth", 40)
KEY = ["name", "ndcg@10", "ndcg@10_ci_low", "ndcg@10_ci_high", "recall@10", "same_subject@10", "diversity", "query_latency_ms"]

board = pd.read_csv(RESULTS_DIR / "leaderboard.csv").sort_values("ndcg@10", ascending=False)
print(f"{len(board)} technique×config rows on the cross-listing board\n")
board[KEY].reset_index(drop=True)

# %% [markdown]
# ## 1. The headline race, with error bars
#
# NDCG@10 per technique with its bootstrap CI. The decisive fact isn't just the
# order — it's how much the **CIs overlap**. Techniques whose intervals overlap are
# statistically a tie, however different their point estimates look.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nbtools import plot_metric_ci

# de-clutter: shorten names and show the spread of distinct families
show = board.drop_duplicates("name").head(12)
labels = [n.split("(")[0] + ("…" if "(" in n else "") for n in show["name"]]
cis = list(zip(show["ndcg@10_ci_low"], show["ndcg@10_ci_high"]))
ax = plot_metric_ci(labels, show["ndcg@10"].tolist(), cis, title="Cross-listing lens · NDCG@10 (95% CI)")
plt.tight_layout()
ax.figure

# %% [markdown]
# ## 2. The cross-listing lens flatters near-duplicates
#
# Cross-listed twins share near-identical text, so this lens is *easy* — even plain
# lexical methods cluster near the top, and the leaders sit inside each other's CIs.
# It validates **correctness**, not quality. Note the `same_subject@10` column: high
# values are a *sanity floor*, not a goal (a subject-only model would max it while
# being useless).

# %%
families = {
    "sbert": "semantic", "rerank": "semantic+rerank", "tfidf": "lexical",
    "bm25": "lexical", "lsa": "topic", "nmf": "topic", "lda": "topic",
    "metadata": "metadata", "llm_tags": "llm",
}
board["family"] = board["name"].str.split("(").str[0].map(families).fillna("other")
board.groupby("family")["ndcg@10"].agg(["max", "count"]).sort_values("max", ascending=False)

# %% [markdown]
# ## 3. The free-text lens — where meaning matters
#
# The judged-query board scores `recommend_by_text` on hand-labeled queries — the
# only lens that measures *free-text* search, the mode semantic and topic methods
# are built to win (and lexical, blind to synonyms, is meant to struggle).

# %%
text_board = pd.read_csv(RESULTS_DIR / "leaderboard_text.csv").sort_values("ndcg@10", ascending=False)
text_board[["name", "ndcg@10", "recall@10", "mrr"]].head(8).reset_index(drop=True)

# %% [markdown]
# ## 4. The graph's own board (not comparable)
#
# The graph reads cross-listings, so it's scored only on **held-out** edges — a
# different task ("recover a removed twin") on a different board. It is **never**
# ranked against the content rungs above; comparing them would be apples to oranges.

# %%
held = pd.read_csv(RESULTS_DIR / "leaderboard_heldout.csv").sort_values("ndcg@10", ascending=False)
held[["name", "ndcg@10", "recall@10", "n_queries"]].reset_index(drop=True)

# %% [markdown]
# ## 5. The story across all eight techniques
#
# | Rung | Notebook | Verdict on the cross-listing lens |
# |---|---|---|
# | Lexical (TF-IDF, BM25) | [01](01_lexical.py) | strong — twins share text; the correctness floor |
# | Topics (LSA/NMF/LDA) | [02](02_topics.py) | ~ties lexical; pays off in interpretability + free text |
# | **Semantic (SBERT)** | [03](03_embeddings.py) | **tops the board**, and actually does free-text search |
# | Rerank + MMR | [04](04_rerank.py) | marginal here (twins already on top); MMR buys diversity |
# | Metadata fusion | [05](05_metadata.py) | *hurts* — structure pulls non-twins up (honest negative) |
# | Graph (PPR) | [06](06_graph.py) | own held-out board; not comparable |
# | Clustering | [07](07_clustering.py) | not a ranker — a diagnostic |
# | LLM | [08](08_llm.py) | lost at ranking, won at *explaining* |
#
# ## 6. Takeaways
#
# - **SBERT wins the primary lens *and* free text** — the one rung strong at both,
#   which is why it's the UI default.
# - **The cross-listing race is tight and CI-bound:** never crown a winner on a
#   sub-CI gap. The interesting results are the *negatives* — metadata fusion
#   hurting, the LLM rungs losing — which a single-lens reading would hide.
# - **Different goals need different boards:** free-text (judged queries) and graph
#   (held-out edges) are separate tasks, scored separately. One number is never the
#   whole story — that's the entire reason the harness reports three lenses.
#
# **Source:** [`scripts/run_eval.py`](../scripts/run_eval.py),
# [`courserec/eval.py`](../src/courserec/eval.py) · regenerate the boards with
# `python scripts/run_eval.py`. **You've reached the end of the path — back to
# [the index](README.md).**
