# ADR-0007: Clustering + 2-D map as a diagnostic, not a recommender (Phase 6)

**Date:** 2026-06-19
**Status:** Accepted

## Context
Phase 6 is Track B rung 7 (plan §2.7 / §5): cluster the embeddings and draw a
2-D map colored by subject. Unlike every prior phase it produces no ranker — the
plan calls it explicitly "diagnostic, not a ranker, but feeds diversity/coverage
analysis." Several decisions were load-bearing.

1. **Where it sits in the architecture.** Every technique so far subclasses
   `Recommender` and is scored by the shared harness. A clustering has no
   `recommend_similar`/`recommend_by_text` and no ground truth to rank against —
   forcing it into the interface would mean inventing a fake ranking.
2. **Dependency budget.** The plan names KMeans/HDBSCAN/UMAP/t-SNE. The repo
   keeps heavy deps behind extras and must run end-to-end with no API key
   (CLAUDE.md, plan §1). UMAP (`umap-learn`) pulls `numba`/`llvmlite`; HDBSCAN was
   historically a separate package.
3. **Where embeddings come from.** Re-encoding 11k courses to cluster them would
   duplicate the SBERT recommender's work and ignore its cache (plan §1, rule 4).
4. **What "coherence" means without a ranking target.** A diagnostic still needs
   numbers, or it is just a picture.
5. **Tractability of the metrics/plot at ~11k points.** Exact silhouette is
   `O(n²)`; t-SNE on the full catalog is the slow step.

## Decision
1. **A plain module, not a `Recommender`.** `src/courserec/cluster.py` exposes
   functions + a `ClusterResult` dataclass; `scripts/run_clustering.py` runs it.
   It writes `results/cluster_report.{md,csv}` and `results/plots/embedding_map.png`
   and **never touches any leaderboard**. No fake interface, no leaderboard row.
2. **scikit-learn only for clustering — zero new required deps.** KMeans,
   `AgglomerativeClustering` (Ward), and `sklearn.cluster.HDBSCAN` (built in since
   sklearn 1.3) all ship in the base install. No `hdbscan` package needed.
3. **Reuse the cached SBERT vectors.** `load_sbert_embeddings` fits an
   `SbertRecommender` (which reloads its artifact — no re-encode) and reads new
   read-only `embeddings`/`course_ids` properties added to `_EmbeddingRecommender`.
   The diagnostic consumes vectors someone else already paid to compute.
4. **`viz` extra + graceful projector fallback.** `matplotlib` and `umap-learn`
   go in a new optional `viz` extra. `project_2d` prefers UMAP and **falls back to
   scikit-learn's t-SNE** when it is absent; `plot_projection` skips (and flags)
   the figure if matplotlib is missing. So clustering + metrics run on base +
   `semantic`; only the picture needs `viz` — never a hard failure (plan §1).
5. **Two coherence axes + degeneracy guards.** Report **silhouette** (internal,
   cosine, sampled at 2,000 for tractability) and **subject purity** (external:
   size-weighted dominant-subject share — does text recover subjects with no
   metadata?), plus **largest-cluster fraction** and **noise count** so a single
   giant blob or all-noise HDBSCAN reads as low coherence rather than a high score.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Make clustering a `Recommender` (recommend cluster-mates) | One interface; joins the leaderboard | No ground-truth ranking target; cluster-mate ≠ relevant; the plan says diagnostic | Would fabricate a ranking the data can't validate |
| Require `umap-learn` | Faster, more global-structure-faithful map | Heavy `numba`/`llvmlite` deps; breaks no-key end-to-end if uninstalled | t-SNE fallback gives a valid plan §2.7 map with zero required deps |
| Separate `hdbscan` package | Mature, more knobs (e.g. soft clustering) | New dependency for what sklearn now ships | `sklearn.cluster.HDBSCAN` covers the diagnostic need |
| Re-embed the catalog inside cluster.py | Self-contained | Duplicates encoding, ignores the artifact cache (rule 4) | Reusing the SBERT artifact is the whole caching design |
| Exact (unsampled) silhouette | A single precise number | `O(n²)` time/memory at 11k | Sampled estimate is enough for a shape probe |

## Consequences
**Positive:** Phase 6 lands the map + coherence table with **zero new required
dependencies** (sklearn-only clustering; `viz` optional), reuses the cached
embeddings, and stays cleanly outside the leaderboard. The read-only
`embeddings`/`course_ids` accessors give future diagnostics the same clean hook.

**Negative / honest finding:** the diagnostic's value is interpretive, not a
metric that moves a ranking — silhouette is low (~0.08–0.12 under forced k=100)
and HDBSCAN calls ~90% of the catalog noise, so the headline is "the SBERT space
is a smooth manifold, not tidy clusters." Subject purity ~0.32 (no metadata used)
confirms coherent-but-blended neighborhoods. This is the point — it explains why
semantic similarity is graded rather than categorical — but it produces no
leaderboard win. See `docs/RESULTS.md` Phase 6.

**Neutral:** cluster count (k=100) and `min_cluster_size` are chosen knobs, not
tuned; the numbers shift with them. The map here is t-SNE (UMAP not installed),
which is slower and less global-structure-faithful than UMAP would be.

## Implementation notes
`src/courserec/cluster.py`: `ClusterResult`, `load_sbert_embeddings`,
`cluster_embeddings`, `evaluate_clustering`, `run_clustering`, `_subject_purity`,
`_silhouette`, `project_2d`, `plot_projection`, `report_markdown`. Driver:
`scripts/run_clustering.py`. Read-only `embeddings`/`course_ids` properties added
to `_EmbeddingRecommender` in `src/courserec/recommenders/embeddings.py`. `viz`
extra + `PLOTS_DIR` added in `pyproject.toml` / `src/courserec/config.py`.
Synthetic-blob contract tests in `tests/test_cluster.py`. Builds on the semantic
rung of [ADR-0004](0004-semantic-vectors.md).
