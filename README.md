# course-rec-lab

A sandbox for learning recommender systems by implementing, documenting, and
comparing **content-based** techniques on the UC Berkeley course catalog
(~11,073 courses after cleaning). No user-interaction data exists, so
collaborative filtering is out of scope. Every technique conforms to one
interface, is scored by one shared evaluation harness, and ranked on a
leaderboard.

The full contract is [docs/roadmap/recommender_plan.md](docs/roadmap/recommender_plan.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The raw catalog CSV lives at `data/raw/courses-report_2026-06-02.csv` (gitignored
along with everything under `data/` — keep your own copy).

## Run

> Full operational runbook — every script + flag, every model, install tiers,
> artifacts, and troubleshooting — in **[docs/RUNBOOK.md](docs/RUNBOOK.md)**.

```bash
python scripts/prepare_data.py   # raw CSV -> data/processed/courses.parquet
python scripts/enrich_catalog.py # Phase 7: LLM-tag the eval subset via Ollama (--all for full)
python scripts/run_eval.py       # fit + score all techniques -> results/leaderboard.{md,csv}
python scripts/run_clustering.py # Phase 6 diagnostic -> results/cluster_report.md + plots/
python scripts/explain_recs.py --seed "COMPSCI 189"  # Phase 7c: "why this fits" lines via Ollama
streamlit run app/streamlit_app.py  # Phase 8 UI: Explore / Compare / Leaderboard / Map
pytest                           # tests
ruff check . && black .          # lint / format
```

Or run the UI as a warm, offline Docker image (catalog + artifacts + MiniLM baked in,
no first-load encode, CPU-only — [ADR-0013](docs/adr/0013-deploy-warm-docker-image.md)):

```bash
docker build -t course-rec-ui . && docker run --rm -p 8501:8501 course-rec-ui  # http://localhost:8501
```

The build `COPY`s the (gitignored) processed catalog + artifacts, so build from a repo
that has already run the pipeline + eval. See the RUNBOOK's **Deploy** section.

**Learning the techniques?** [`notebooks/`](notebooks/) has ten step-by-step
breakdowns ([ADR-0014](docs/adr/0014-teaching-notebooks.md)) — one per technique
family, each building the method from primitives on the real catalog and running its
eval live, plus a data/eval foundation (`00`) and a cross-technique synthesis (`09`):

```bash
pip install -e ".[notebooks,semantic]"
jupytext --to ipynb notebooks/01_lexical.py && jupyter lab notebooks/01_lexical.ipynb
```

The semantic rung needs `pip install -e ".[semantic]"`; the Phase 6 map needs
`".[viz]"` (matplotlib; UMAP optional — t-SNE fallback otherwise); the Phase 8 UI
needs `".[ui,semantic]"` (add `viz` for the Map view's faster UMAP projection, else
it falls back to t-SNE). The Phase 7 LLM rung needs a local
[Ollama](https://ollama.com) daemon + a pulled model (no API key, no extra Python
deps); `run_eval` skips it gracefully when Ollama is absent, and the UI's opt-in
"why this fits" column simply stays blank.

## Status

**Phases 0–8 — complete.** On the Phase 0 pipeline (cleaned catalog of 11,073
unique courses), the techniques score through one shared eval harness
([src/courserec/eval.py](src/courserec/eval.py)) on **three lenses** —
cross-listing twins (automatic), a hand-labeled judged free-text set (44
paraphrase-extreme queries), and a held-out cross-listing edge split for the
graph — with full metrics and bootstrap CIs, written to one-command leaderboards
([leaderboard.md](results/leaderboard.md), `leaderboard_text.md`,
`leaderboard_heldout.md`):

- **Phase 1 — lexical:** TF-IDF+cosine, Okapi BM25 ([lexical.py](src/courserec/recommenders/lexical.py)).
- **Phase 2 — topic models:** LSA, NMF, LDA ([topics.py](src/courserec/recommenders/topics.py)).
- **Phase 3 — semantic vectors:** local SBERT (MiniLM/MPNet) + an API backend that
  skips with no key ([embeddings.py](src/courserec/recommenders/embeddings.py)).
- **Phase 4 — retrieve→rerank→MMR:** cross-encoder rerank with an MMR diversity
  knob ([rerank.py](src/courserec/recommenders/rerank.py)).
- **Phase 5 — course graph (PPR):** personalized-PageRank over cross-listing +
  subject/dept aux nodes, scored only on a held-out edge split
  ([graph.py](src/courserec/recommenders/graph.py)).
- **Phase 5 — metadata fusion:** one-hot subject+dept+level+units fused with
  TF-IDF under a weight λ — a clean ablation that *loses* on cross-listing
  ([metadata.py](src/courserec/recommenders/metadata.py)).
- **Phase 6 — clustering + 2-D map:** KMeans / Ward / HDBSCAN over the SBERT
  vectors + a subject-colored map — a **diagnostic**, not a ranker
  ([cluster.py](src/courserec/cluster.py)).
- **Phase 7 — LLM enrichment (tag rung):** local Ollama (qwen3:8b, no API key)
  extracts per-course tags, ranked by TF-IDF cosine over tag profiles
  ([llm.py](src/courserec/recommenders/llm.py)).
- **Phase 7b — zero-shot LLM reranker:** SBERT retrieves top-20, qwen3:8b reorders
  them over their *full* text in one deterministic call (cached); **measured — does
  not beat the base** (a second honest negative result)
  ([llm.py](src/courserec/recommenders/llm.py), [ADR-0010](docs/adr/0010-llm-reranker.md)).
- **Phase 7c — "why this fits" explainer:** qwen3:8b writes a one-line justification
  for an SBERT recommendation — a UI helper, **not** a scored recommender (closes
  Track B.8). The two ranking failures pointed here: the LLM earns its cost
  *explaining* a ranking, not producing one
  ([llm.py](src/courserec/recommenders/llm.py), [ADR-0011](docs/adr/0011-llm-explainer.md)).
- **Phase 8 — minimal Streamlit UI:** four views — Explore (course or free-text →
  top-k with scores + an opt-in "why this fits" column), Compare (one query, two
  techniques), Leaderboard (the eval table + map), and a live **Map** (an interactive
  2-D projection where a seed + its SBERT recommendations light up). Testable,
  Streamlit-free modules — a technique registry ([app/registry.py](app/registry.py)),
  an explanatory glossary ([app/glossary.py](app/glossary.py): per-technique blurbs,
  per-metric definitions, the eval lenses), and a cached projection
  ([app/projection.py](app/projection.py)) — feed a thin view layer
  ([app/streamlit_app.py](app/streamlit_app.py)). The UI exposes a fast offline
  subset of rungs while the full sweep stays in the leaderboard, with a `family`
  column and hover-for-definition tooltips so the cryptic rows read clearly
  ([ADR-0012](docs/adr/0012-streamlit-ui.md)).

**Headline:** SBERT MiniLM tops both ranking lenses and beats the best lexical
config on free text *decisively* (NDCG@10 0.682 vs 0.499, non-overlapping CIs).
Honest findings: the graph recovers only ~23% of held-out twins because
near-identical twin text gives content methods everything
([ADR-0006](docs/adr/0006-graph-heldout.md)); the embedding space is a smooth
manifold, not tidy clusters ([ADR-0007](docs/adr/0007-clustering-diagnostic.md));
metadata fusion *hurts* the cross-listing target because 99.7% of twins span
subjects ([ADR-0008](docs/adr/0008-metadata-fusion.md)); and the LLM tag rung
*looked* like it beat every lexical baseline on the 12.5% eval-target subset, but
**full-catalog (100%) enrichment overturned it** — at full coverage it ties lexical
on cross-listing (0.957) and falls *below* plain TF-IDF on free text (0.404),
because distilling a description to ~6–12 tags loses more signal than the LLM's
normalization adds ([ADR-0009](docs/adr/0009-llm-enrichment-ollama.md)); and the
zero-shot LLM reranker **also fails to beat the SBERT base** (xlist 0.965 vs 0.971,
text 0.656 vs 0.682 — both within CIs, recall@10 dips) because SBERT's top-20 is
already near-ceiling, so reordering it has no headroom and costs ~4 s/query
([ADR-0010](docs/adr/0010-llm-reranker.md)). See
[docs/RESULTS.md](docs/RESULTS.md) and [docs/TRADEOFFS.md](docs/TRADEOFFS.md).
The "why this fits" explainer (Phase 7c) lands the LLM where it pays off —
*justifying* an SBERT ranking, not producing one — and closes Track B.8
([ADR-0011](docs/adr/0011-llm-explainer.md)). The **Phase 8 Streamlit UI** now
surfaces all of it — techniques, the explainer line, and the leaderboard — from one
`streamlit run` ([ADR-0012](docs/adr/0012-streamlit-ui.md)). The swappable
`Recommender` interface is in
[src/courserec/interfaces.py](src/courserec/interfaces.py).

See [CHANGELOG.md](CHANGELOG.md) for history and [docs/adr/](docs/adr/) for
architecture decisions.
