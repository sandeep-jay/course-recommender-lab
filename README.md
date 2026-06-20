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

```bash
python scripts/prepare_data.py   # raw CSV -> data/processed/courses.parquet
python scripts/enrich_catalog.py # Phase 7: LLM-tag the eval subset via Ollama (--all for full)
python scripts/run_eval.py       # fit + score all techniques -> results/leaderboard.{md,csv}
python scripts/run_clustering.py # Phase 6 diagnostic -> results/cluster_report.md + plots/
pytest                           # tests
ruff check . && black .          # lint / format
```

The semantic rung needs `pip install -e ".[semantic]"`; the Phase 6 map needs
`".[viz]"` (matplotlib; UMAP optional — t-SNE fallback otherwise). The Phase 7 LLM
rung needs a local [Ollama](https://ollama.com) daemon + a pulled model (no API
key, no extra Python deps); `run_eval` skips it gracefully when Ollama is absent.

## Status

**Phases 0–7 (tag rung) — complete.** On the Phase 0 pipeline (cleaned catalog of 11,073
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
  them over their *full* text in one deterministic call (cached); built +
  live-verified, leaderboard delta pending
  ([llm.py](src/courserec/recommenders/llm.py), [ADR-0010](docs/adr/0010-llm-reranker.md)).

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
normalization adds ([ADR-0009](docs/adr/0009-llm-enrichment-ollama.md)). See
[docs/RESULTS.md](docs/RESULTS.md) and [docs/TRADEOFFS.md](docs/TRADEOFFS.md).
Next: measure the just-built zero-shot LLM reranker (run `scripts/run_eval.py` with
Ollama up to fill its rerank cache), then the "why this fits" explanation over full
candidate text (Phase 7 b/c). The swappable `Recommender` interface is in
[src/courserec/interfaces.py](src/courserec/interfaces.py).

See [CHANGELOG.md](CHANGELOG.md) for history and [docs/adr/](docs/adr/) for
architecture decisions.
