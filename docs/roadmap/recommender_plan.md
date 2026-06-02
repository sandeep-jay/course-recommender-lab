# course-rec-lab — Build Plan

A sandbox for learning recommender systems by implementing, documenting, and
comparing **content-based** techniques on a university course catalog. No user
interaction data exists, so collaborative filtering is explicitly out of scope.
Every technique is scored by one shared evaluation harness and ranked on a
leaderboard, with a minimal UI for inspection.

This document is the contract. Build it in phases, commit per phase, and make
each technique conform to one interface so they are swappable and comparable.

---

## 0. The dataset (already profiled — do not re-discover)

Input file: `data/raw/courses-report_2026-06-02.csv` (UC Berkeley course catalog).

- ~11,091 course rows (the file has ~12,144 physical lines; the difference is
  RFC-4180 quoted newlines embedded in description fields — parse with pandas,
  never `head`/`split` by line).
- 242 subjects, 113 departments.
- The string `"-"` is the catalog's **null token**. Replace it with real NA on load.

Column usefulness (populated %):

| Column | Populated | Role |
|---|---|---|
| Subject | 100% | metadata / facet / filter |
| Course Number | 100% | parse to **level** (1-99 lower-div, 100-199 upper-div, 200+ grad) |
| Department(s) | 100% | metadata / facet |
| Course Title | 100% | **primary text signal** |
| Course Description | 97.9% | **primary text signal** (median ~54 words, min 1, max 181) |
| Cross-Listed Course(s) | 9.8% | **GROUND TRUTH for evaluation** — do not use as a model feature |
| Repeat Rules | 97.5% | mostly boilerplate; strip |
| Credits min/max | 100% | metadata / filter (units) |
| Terms Offered | 0% | DEAD — drop |
| Offering Information / Details / Additional | ~0–1% | DEAD — drop |
| Repeat Rule: Special Circumstances | 0.5% | DEAD — drop |

A stable `course_id` must be synthesized as `f"{Subject} {Course Number}"`
(e.g. `AEROENG 1`). Some courses have 1-word or missing descriptions — handle
sparse-text items gracefully (fall back to title; never crash).

---

## 1. Architecture

### Swappable interface (the load-bearing decision)

Every technique implements one abstract base class so the eval harness and UI
treat them identically.

```python
# src/courserec/interfaces.py
from dataclasses import dataclass

@dataclass
class Rec:
    course_id: str
    score: float

class Recommender(ABC):
    name: str                      # unique, used in leaderboard
    config: dict                   # hyperparams, logged with results

    def fit(self, courses: "pd.DataFrame") -> None: ...
    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]: ...
    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]: ...
        # may raise NotImplementedError for techniques that are item-to-item only
```

Rules:
- `recommend_similar` must **exclude the seed course itself** from results.
- Models persist fitted artifacts to `artifacts/<name>/` and load if present
  (never recompute embeddings on every run).
- No technique may read `Cross-Listed Course(s)` as an input feature — that
  column is the evaluation target and using it is leakage. (The graph technique
  is the one exception and must be evaluated only on a held-out split of edges.)

### Repo layout

```
course-rec-lab/
  README.md
  PLAN.md                      # this file
  pyproject.toml               # pinned deps
  data/raw/ data/processed/
  artifacts/                   # fitted models, embedding caches (gitignored)
  src/courserec/
    data.py                    # load, clean, build text field, parse level
    interfaces.py
    eval.py                    # ground truth, metrics, harness
    cluster.py                 # KMeans / HDBSCAN / UMAP
    recommenders/
      lexical.py topics.py embeddings.py rerank.py metadata.py graph.py llm.py
  scripts/
    prepare_data.py            # raw csv -> processed parquet
    run_eval.py                # fit all, score all, write leaderboard
    build_judged_queries.py    # helper to assemble the text-query test set
  app/streamlit_app.py
  results/leaderboard.md results/leaderboard.csv results/plots/
  tests/
  docs/TRADEOFFS.md docs/RESULTS.md docs/METHODOLOGY.md
```

### Environment & reproducibility

- Pin all deps in `pyproject.toml`. Set a global seed (`RANDOM_SEED = 42`).
- Embedding cache: key by `sha1(model_name + normalized_text)`, store as
  `.npz`/parquet in `artifacts/`. API-embedding runs must log token count + cost.
- `.gitignore`: `artifacts/`, `*.npz`, `data/processed/`, `.env`.
- API keys via env vars only; the repo must run end-to-end through Phase 6 with
  **no API key** (local-only). LLM/API-embedding phases degrade gracefully if
  the key is absent (skip + note in leaderboard).

---

## 2. Techniques to implement

### Track A — similarity ladder (compare in order, simple → frontier)

1. **Lexical** (`lexical.py`): TF-IDF + cosine, and BM25 (Okapi). Run a small
   preprocessing sweep as configs: stopwords on/off, lemmatization on/off,
   title-weight multiplier, 1- vs 1–2-grams.
2. **Latent topics** (`topics.py`): LSA (TruncatedSVD on TF-IDF), NMF, LDA.
   Recommend by similarity in topic space. Persist topic-term tables for
   interpretation.
3. **Semantic vectors** (`embeddings.py`): SBERT (`all-MiniLM-L6-v2` and one
   larger model) locally; then an API embedding model behind the same interface.
   Use FAISS/hnswlib for ANN (overkill at 11k but part of the learning).
4. **Retrieve + rerank** (`rerank.py`): bi-encoder retrieves top-50, cross-encoder
   reranks. Add MMR for diversity (tunable λ).

### Track B — complementary methods

5. **Metadata fusion** (`metadata.py`): one/multi-hot of subject, department,
   level, units; concatenate (weighted) with a text vector; tune weighting.
6. **Course graph** (`graph.py`): build a graph from cross-listings + shared
   subject/dept; learn node2vec embeddings; recommend by graph proximity.
   **Held-out edge split required** (see leakage rule).
7. **Clustering + map** (`cluster.py`): KMeans, agglomerative, HDBSCAN over
   embeddings; UMAP/t-SNE 2-D plot colored by subject. Diagnostic, not a ranker,
   but feeds diversity/coverage analysis.
8. **LLM enrichment** (`llm.py`): (a) LLM extracts structured tags (topics,
   skills, level, prereqs-mentioned) per course → richer features; (b) zero-shot
   LLM reranker over candidate sets; (c) generate a one-line "why this fits"
   explanation for the UI.

---

## 3. Evaluation methodology (the core of the project)

Because there are no clicks or ratings, ground truth is constructed. Use
**three independent lenses** — no single one is sufficient (see §6).

1. **Cross-listing pairs (primary, automatic):** a course and its cross-listed
   twin should rank each other near the top. Measure on the ~10% of courses that
   have cross-listings. Caveat: twins have near-identical text, so lexical
   methods will nail this trivially — it validates correctness more than quality.
2. **Same-subject coherence (weak proxy, automatic):** fraction of top-k sharing
   the seed's subject. Useful as a sanity floor, but a model that *only* returns
   same-subject courses scores high here while being useless — report it, do not
   optimize for it.
3. **Judged text-query set (for `recommend_by_text`, semi-manual):** hand-build
   ~20–30 natural queries ("practical deep learning", "ethics of technology",
   "intro statistics for social science") with a few relevant `course_id`s each.
   This is the **only** way to evaluate the free-text mode — flag it as a gap if
   skipped. Optionally augment with an **LLM-as-judge** that rates relevance of
   top-k; validate the judge against the human labels before trusting it.

Metrics (`eval.py`): Recall@k, Precision@k, MRR, MAP, NDCG@k (k ∈ {5,10,20}),
plus **catalog coverage**, **intra-list diversity**, and **novelty**. Report
**bootstrap confidence intervals** on the primary metric — the ground-truth set
is small, so metric differences need error bars before you call a winner.

Leakage discipline: when cross-listings are the target, no model may use that
column as a feature. The graph model evaluates only on held-out edges.

Leaderboard (`run_eval.py`): writes `results/leaderboard.{md,csv}` — one row per
technique×config with all metrics, fit time, query latency, and (if applicable)
API cost. Sort by NDCG@10 by default; keep it regenerable in one command.

---

## 4. Minimal UI (`app/streamlit_app.py`)

Keep it to three things:
- **Explore:** pick a course (or type a free-text query) + choose a technique →
  show top-k with scores and the "why this fits" line (if LLM enrichment ran).
- **Compare:** same query, two techniques side by side.
- **Leaderboard:** render `results/leaderboard.csv` as a sortable table + the
  UMAP plot.

No auth, no database, no styling beyond Streamlit defaults.

---

## 5. Phased milestones (build + commit in this order)

Each phase has an acceptance test; do not advance until it passes.

- **Phase 0 — Scaffold & data.** Repo, deps, `prepare_data.py`, `interfaces.py`,
  tests for the loader (null handling, id synthesis, level parsing, row count).
  *Accept:* `pytest` green; processed parquet has ~11,091 rows, 0 `"-"` values.
- **Phase 1 — Lexical + harness + leaderboard.** TF-IDF & BM25, full `eval.py`,
  cross-listing + same-subject lenses, first leaderboard.
  *Accept:* one command produces a leaderboard with ≥2 rows and CIs.
- **Phase 2 — Topic models.** LSA/NMF/LDA + interpretation tables.
- **Phase 3 — Semantic.** SBERT (+caching, FAISS), then API embeddings behind
  the same interface; cost logged. *Accept:* runs local-only with no API key.
- **Phase 4 — Rerank + MMR.** Two-stage pipeline; diversity metric moves with λ.
- **Phase 5 — Metadata fusion + graph.** Held-out edge eval for graph.
- **Phase 6 — Clustering + UMAP.** Plot saved to `results/plots/`.
- **Phase 7 — LLM enrichment + LLM-as-judge.** Judge validated vs human labels.
- **Phase 8 — Streamlit UI.** Explore / Compare / Leaderboard.
- **Cross-cutting (every phase):** docstrings (idea, math sketch, complexity,
  when it wins/loses), update `docs/TRADEOFFS.md` and `docs/RESULTS.md`.

`docs/TRADEOFFS.md` is a matrix: technique × {quality, speed, interpretability,
cost, cold-start behavior, code complexity, when to prefer it}.

---

## 6. Known gaps & risks (read before starting)

These are the project's real blind spots — design around them, don't discover
them late.

1. **"Similar" ≠ "good recommendation."** Pure text similarity returns the most
   *redundant* course, not necessarily a useful *next* or *complementary* one.
   Decide up front what a recommendation is for (alternative / next-in-sequence /
   complementary) — course level and prereq mentions can encode progression.
2. **Cross-listing ground truth is necessary but insufficient and partly
   trivial.** It covers only ~10% of courses, rewards near-duplicate text, and
   tests only item-to-item. Hence the judged query set and diversity/coverage
   metrics — without them you'll "win" while recommending nothing useful.
3. **Free-text mode has no automatic ground truth.** It needs the hand-built
   judged query set (and/or a validated LLM judge). Don't ship the UI's query box
   without some way to evaluate it.
4. **Statistical significance.** Small ground-truth set → report bootstrap CIs;
   don't crown a winner on a 0.01 NDCG gap.
5. **Subject imbalance** (EDUC 345 vs tiny subjects) can bias methods toward
   large subjects; coverage/novelty metrics surface this.
6. **Cost & caching for embeddings/LLM.** Re-embedding every run is slow and (for
   APIs) expensive — caching and cost logging are requirements, not nice-to-haves.
7. **Reproducibility & artifact bloat.** Seeds, pinned deps, gitignored vectors.

---

## 7. First move

Phase 0 + Phase 1 give a working, comparable system from the first commit:
clean data pipeline, the swappable interface, TF-IDF/BM25 baselines, the
cross-listing evaluator, and a leaderboard. Everything after that slots into the
same harness.