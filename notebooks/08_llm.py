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
# # 08 · LLM enrichment — tags, reranking, explanations
#
# > Assumes [00](00_data_and_eval.py). Uses a **local** LLM via
# > [Ollama](https://ollama.com) (`qwen3:8b`) — **no API key**. Every LLM call here
# > degrades gracefully: if the daemon is down, the notebook still runs from cache.
#
# Three ways an LLM can help a content recommender, each tested in the repo:
# **(a) extract structured tags** per course → richer features; **(b) zero-shot
# rerank** a candidate set; **(c) explain** why a recommendation fits. The honest
# punchline up front: on *ranking*, the local LLM **lost** to plain SBERT — but it
# earns its keep at **explaining**. Let's see why.

# %%
import numpy as np
import pandas as pd

from courserec.data import load_processed
from courserec.recommenders.llm import OllamaClient

pd.set_option("display.max_colwidth", 60)
courses = load_processed()
seed = "COMPSCI 189"

client = OllamaClient()  # localhost:11434, qwen3:8b
ollama_up = client.available()
print(f"Ollama available: {ollama_up}  (live calls run if True, else cache-only)")

# %% [markdown]
# ## 1. (a) Extract structured tags from one course
#
# **The transformation.** Feed a course's title + description to the LLM with a
# fixed JSON schema, and it returns `topics / skills / level / prereqs_mentioned`
# — a distilled, vocabulary-normalized profile that raw text doesn't give you.
# (Deterministic: `temperature=0`, fixed seed.)

# %%
if ollama_up:
    tags = client.extract_tags(courses.loc[seed, "title"], str(courses.loc[seed, "text"]))
    print(f"{seed} — {courses.loc[seed, 'title']}")
    print(f"  topics : {tags.topics}")
    print(f"  skills : {tags.skills}")
    print(f"  level  : {tags.level}")
    print(f"\n  profile_text (what the tag rung indexes):\n  {tags.profile_text()}")
else:
    print("Ollama down — skipping the live extraction; the rung below reads the tag cache.")

# %% [markdown]
# ## 2. (a) The tag rung — rank by TF-IDF over tag profiles
#
# `LLMTagRecommender` builds a TF-IDF space over the **tag profiles** (cached by
# `enrich_catalog.py`) instead of raw text. Courses without cached tags fall back
# to raw text, so it never crashes. `fit` reads the cache only — no LLM calls.

# %%
from nbtools import recs_to_frame

from courserec.recommenders.llm import LLMTagRecommender

tag_rec = LLMTagRecommender()
tag_rec.fit(courses)  # warm: reads the tag cache / cached artifact
recs_to_frame(tag_rec.recommend_similar(seed, k=8), courses)

# %% [markdown]
# ## 3. Evaluate the tag rung live (sample) — and why it lost
#
# Score the tag rung against the SBERT base on a reproducible sample of seeds.

# %%
from courserec.eval import build_crosslist_truth, build_reference_space, score_crosslist
from courserec.recommenders.embeddings import SbertRecommender

truth = build_crosslist_truth(courses)
rng = np.random.default_rng(42)
sample = {s: truth[s] for s in rng.choice(sorted(truth), size=120, replace=False)}
reference = build_reference_space(courses)

sbert = SbertRecommender(model_name="all-MiniLM-L6-v2")
sbert.fit(courses)
for label, rec in [("SBERT base", sbert), ("LLM tag rung", tag_rec)]:
    r = score_crosslist(rec, courses, sample, reference, n_boot=200)
    print(f"{label:14s}  NDCG@10={r.metrics['ndcg@10']:.4f}  (CI {r.ndcg10_ci[0]:.3f}–{r.ndcg10_ci[1]:.3f})  n={r.n_queries}")

# %% [markdown]
# The tag rung sits **below** SBERT: distilling a course to a handful of tags
# *throws away* signal that the full-text embedding keeps, and on the near-identical
# cross-listing twins that loss shows. The zero-shot LLM **reranker** (ADR-0010)
# told the same story — reordering SBERT's candidates didn't beat SBERT's own order.

# %% [markdown]
# ## 4. (c) Where the LLM wins — explaining a recommendation
#
# The one place the local LLM clearly earns its cost: **not** ranking, but writing a
# one-line "why this fits" for a recommendation a stronger method already produced.
# `RecommendationExplainer` is deliberately **not** a `Recommender` — it has no
# ranking to score; it's the UI's explanation layer. Degrades to `None` if Ollama
# is down.

# %%
from courserec.recommenders.llm import RecommendationExplainer

explainer = RecommendationExplainer().fit(courses)
top = sbert.recommend_similar(seed, k=3)
print(f"Why these fit {seed} ({courses.loc[seed, 'title']}):\n")
for r in top:
    why = explainer.explain_seed(seed, r.course_id)  # cache → live → None
    print(f"• {r.course_id} — {courses.loc[r.course_id, 'title']}")
    print(f"    {why or '(no explanation — Ollama unavailable and uncached)'}")

# %% [markdown]
# ## 5. Takeaways
#
# - **LLM as a *feature* / *reranker* lost** (§3): compressing a course to tags, or
#   asking a zero-shot model to reorder, both scored **below** the SBERT base on the
#   cross-listing lens. Distillation discards signal the embedding keeps.
# - **LLM as an *explainer* won** (§4): generating a faithful one-line justification
#   for an already-good recommendation is the task it's actually suited to — and it
#   powers the UI's "why this fits" line.
# - **All local, all optional:** Ollama, no API key, every call degrading to cache
#   or `None` — the suite never hard-fails when the daemon is down (the project's
#   graceful-degradation rule).
#
# **Source:** [`courserec/recommenders/llm.py`](../src/courserec/recommenders/llm.py)
# · **ADRs:** [0009](../docs/adr/0009-llm-enrichment-ollama.md),
# [0010](../docs/adr/0010-llm-reranker.md), [0011](../docs/adr/0011-llm-explainer.md)
# · **Next:** [09 · The leaderboard](09_leaderboard.py).
