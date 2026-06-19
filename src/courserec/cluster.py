"""Clustering + 2-D map over the SBERT embeddings — a diagnostic, not a ranker.

This is Track B rung 7 (recommender_plan.md §2.7 / §5 Phase 6). Unlike every
other technique it does **not** subclass :class:`~courserec.interfaces.Recommender`
and never joins the leaderboard: it answers *shape* questions, not *ranking* ones.
Given the dense semantic vectors the SBERT recommender already cached, it asks —
do the embeddings organize into coherent groups, and do those groups line up with
the catalog's own structure (subjects)? The two outputs are (a) a small table of
cluster-coherence numbers per algorithm and (b) a 2-D map of the whole catalog
colored by subject, which feeds the coverage/diversity story the rankers can only
report as scalars.

What it computes
----------------
*Three clusterings* over the same L2-normalized embeddings:

* **KMeans** — partitions into a fixed ``k`` by minimizing within-cluster
  variance. On unit vectors squared-Euclidean is a monotone function of cosine,
  so this is spherical k-means in effect.
* **Agglomerative (Ward)** — bottom-up merging; gives a different, hierarchy-
  flavored partition at the same ``k`` for contrast.
* **HDBSCAN** — density-based; chooses its own cluster count and labels
  low-density points as **noise** (``-1``). The honest counterpoint to the two
  forced-``k`` methods: it can say "this region has no cluster".

*Coherence metrics* per clustering:

* **Silhouette (cosine, sampled)** — internal validity: how much tighter a point
  sits with its own cluster than the next-nearest one, in ``[-1, 1]``. Sampled
  (``SILHOUETTE_SAMPLE_SIZE``) because the exact score is ``O(n²)``.
* **Subject purity** — external validity against the catalog's own labels: the
  size-weighted mean, over clusters, of the dominant subject's share. High purity
  means clusters recover subjects from text alone (no metadata was used).
* **Largest-cluster fraction** and **noise fraction** — degeneracy guards: one
  giant blob, or HDBSCAN calling everything noise, both read as low coherence.

Math sketch
-----------
Silhouette for point ``i``: ``s(i) = (b − a) / max(a, b)`` where ``a`` is its mean
intra-cluster distance and ``b`` the mean distance to the nearest *other* cluster;
we report the mean over a random sample. Subject purity for cluster ``C`` is
``max_s |{i ∈ C : subject(i) = s}| / |C|``, averaged over clusters weighted by
``|C|``. The 2-D map is UMAP (or t-SNE) — a non-linear projection that preserves
local neighborhoods, so visual proximity ≈ embedding proximity.

Complexity
----------
KMeans is ``O(n · k · d · iters)``; Ward agglomerative is ``~O(n²)`` time and
memory (fine at ~11k on this machine); HDBSCAN is ``~O(n log n)`` after a
neighbor graph. Silhouette is ``O(m²·d)`` on the ``m``-point sample. The 2-D
projection dominates wall-clock (t-SNE on the full catalog is the slow step).

Graceful degradation
--------------------
The clusterings and metrics need only scikit-learn (base install) plus the cached
SBERT vectors (``semantic`` extra). The **plot** needs ``matplotlib`` and the
projector prefers ``umap-learn`` but falls back to scikit-learn's t-SNE; when
matplotlib is absent the numbers still compute and the plot step is skipped and
flagged — never a hard failure (plan §1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from courserec.config import PLOTS_DIR, RANDOM_SEED
from courserec.recommenders.embeddings import SbertRecommender

logger = logging.getLogger(__name__)

#: Default fixed cluster count for KMeans / agglomerative. The catalog has 242
#: subjects; ~100 groups is coarser than per-subject, so a coherent cluster spans
#: related subjects rather than memorizing one — the interesting diagnostic regime.
DEFAULT_N_CLUSTERS: int = 100

#: HDBSCAN's smallest admissible cluster; smaller groups dissolve into noise.
DEFAULT_MIN_CLUSTER_SIZE: int = 15

#: Sample size for the (otherwise O(n²)) silhouette score. Capped at the catalog.
SILHOUETTE_SAMPLE_SIZE: int = 2000

#: Distinct subjects to color individually in the 2-D map; the rest fold into a
#: single muted "other" so the legend stays legible.
PLOT_TOP_SUBJECTS: int = 15

#: t-SNE neighborhood size (only used on the no-UMAP fallback path).
TSNE_PERPLEXITY: float = 30.0

#: HDBSCAN's sentinel for an unclustered (low-density) point.
_NOISE_LABEL: int = -1


@dataclass(frozen=True)
class ClusterResult:
    """One clustering of the embeddings plus its coherence diagnostics.

    Attributes:
        algorithm: Algorithm name (``"kmeans"``, ``"agglomerative"``, ``"hdbscan"``).
        config: The hyperparameters used (logged alongside the numbers).
        labels: Per-course integer cluster id (``-1`` is HDBSCAN noise), row-aligned
            with the embedding matrix / ``course_ids`` it was fit on.
        n_clusters: Number of clusters found, excluding the noise label.
        n_noise: Count of points labeled noise (always 0 for KMeans/agglomerative).
        silhouette: Sampled cosine silhouette in ``[-1, 1]``; ``nan`` when undefined
            (fewer than two non-noise clusters).
        subject_purity: Size-weighted mean dominant-subject share over clusters.
        largest_cluster_frac: Largest non-noise cluster as a fraction of the catalog.
    """

    algorithm: str
    config: dict
    labels: np.ndarray = field(repr=False)
    n_clusters: int
    n_noise: int
    silhouette: float
    subject_purity: float
    largest_cluster_frac: float

    def summary_row(self) -> dict:
        """Return a flat, rounded dict for the report table (drops ``labels``)."""
        return {
            "algorithm": self.algorithm,
            "config": ", ".join(f"{k}={v}" for k, v in self.config.items()),
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "silhouette": round(self.silhouette, 4),
            "subject_purity": round(self.subject_purity, 4),
            "largest_cluster_frac": round(self.largest_cluster_frac, 4),
        }


def load_sbert_embeddings(
    courses: pd.DataFrame, *, model_name: str = "all-MiniLM-L6-v2"
) -> tuple[np.ndarray, list[str]]:
    """Load the cached SBERT embedding matrix and its course-id row order.

    Fits a :class:`SbertRecommender`, which reloads the persisted artifact when
    present (the warm path — no re-encoding), then hands back its normalized
    vectors. The diagnostic consumes embeddings someone else paid to compute.

    Args:
        courses: Processed catalog indexed by ``course_id``.
        model_name: SBERT model whose cached embeddings to use.

    Returns:
        A ``(n_courses, d)`` L2-normalized matrix and the matching ``course_id``
        list (row-aligned).

    Raises:
        EmbeddingsUnavailable: If the ``semantic`` extra is not installed — the
            caller (script) catches this to skip and flag, never hard-fail.
    """
    rec = SbertRecommender(model_name=model_name)
    rec.fit(courses)
    return rec.embeddings, rec.course_ids


def _subject_purity(labels: np.ndarray, subjects: np.ndarray) -> float:
    """Size-weighted mean dominant-subject share over non-noise clusters.

    For each cluster, find the most common subject and take its fraction of the
    cluster; average those fractions weighted by cluster size. Equivalently: the
    fraction of all clustered points that sit with their cluster's plurality
    subject. Returns ``0.0`` when nothing is clustered.
    """
    clustered = labels != _NOISE_LABEL
    if not clustered.any():
        return 0.0
    labels, subjects = labels[clustered], subjects[clustered]
    hits = 0
    for cid in np.unique(labels):
        members = subjects[labels == cid]
        # value_counts on the subject slice; max count is the plurality subject.
        _, counts = np.unique(members.astype(str), return_counts=True)
        hits += int(counts.max())
    return hits / len(labels)


def _silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Sampled cosine silhouette; ``nan`` if fewer than two non-noise clusters.

    Noise points are excluded before scoring (they are not a cluster). The sample
    is drawn with the global seed so the number is reproducible.
    """
    from sklearn.metrics import silhouette_score

    mask = labels != _NOISE_LABEL
    sub_labels = labels[mask]
    if len(np.unique(sub_labels)) < 2:
        return float("nan")
    sample = min(SILHOUETTE_SAMPLE_SIZE, int(mask.sum()))
    return float(
        silhouette_score(
            embeddings[mask],
            sub_labels,
            metric="cosine",
            sample_size=sample,
            random_state=RANDOM_SEED,
        )
    )


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    algorithm: str,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, dict]:
    """Cluster the embeddings with one algorithm and return labels + its config.

    Args:
        embeddings: ``(n, d)`` L2-normalized matrix.
        algorithm: ``"kmeans"``, ``"agglomerative"``, or ``"hdbscan"``.
        n_clusters: Cluster count for the two forced-``k`` algorithms.
        min_cluster_size: HDBSCAN's smallest admissible cluster.

    Returns:
        A ``(labels, config)`` pair — integer labels row-aligned with
        ``embeddings`` (``-1`` = HDBSCAN noise) and the hyperparameters used.

    Raises:
        ValueError: If ``algorithm`` is not one of the three supported names.
    """
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        config = {"n_clusters": n_clusters}
    elif algorithm == "agglomerative":
        # Ward needs Euclidean; on unit-norm rows that is monotone in cosine.
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        config = {"n_clusters": n_clusters, "linkage": "ward"}
    elif algorithm == "hdbscan":
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        config = {"min_cluster_size": min_cluster_size}
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r}")

    logger.info(
        "clustering %d vectors with %s (%s)", len(embeddings), algorithm, config
    )
    labels = model.fit_predict(embeddings)
    return labels, config


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    algorithm: str,
    config: dict,
) -> ClusterResult:
    """Bundle a labeling with its coherence diagnostics into a :class:`ClusterResult`.

    Args:
        embeddings: ``(n, d)`` matrix the labels were fit on.
        labels: Per-point cluster ids (``-1`` = noise).
        subjects: Per-point subject string, row-aligned with ``embeddings``.
        algorithm: Algorithm name for the record.
        config: Hyperparameters for the record.

    Returns:
        The populated :class:`ClusterResult`.
    """
    non_noise = labels[labels != _NOISE_LABEL]
    n_clusters = int(len(np.unique(non_noise)))
    n_noise = int((labels == _NOISE_LABEL).sum())
    if len(non_noise):
        _, sizes = np.unique(non_noise, return_counts=True)
        largest_frac = int(sizes.max()) / len(labels)
    else:
        largest_frac = 0.0
    return ClusterResult(
        algorithm=algorithm,
        config=config,
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=_silhouette(embeddings, labels),
        subject_purity=_subject_purity(labels, subjects),
        largest_cluster_frac=largest_frac,
    )


def run_clustering(
    embeddings: np.ndarray,
    subjects: np.ndarray,
    *,
    algorithms: tuple[str, ...] = ("kmeans", "agglomerative", "hdbscan"),
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> list[ClusterResult]:
    """Cluster + evaluate the embeddings under several algorithms.

    Args:
        embeddings: ``(n, d)`` L2-normalized matrix.
        subjects: Per-point subject string, row-aligned with ``embeddings``.
        algorithms: Which algorithms to run.
        n_clusters: Cluster count for the forced-``k`` algorithms.
        min_cluster_size: HDBSCAN's smallest admissible cluster.

    Returns:
        One :class:`ClusterResult` per algorithm, in the given order.
    """
    results = []
    for algo in algorithms:
        labels, config = cluster_embeddings(
            embeddings,
            algorithm=algo,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )
        results.append(
            evaluate_clustering(
                embeddings, labels, subjects, algorithm=algo, config=config
            )
        )
    return results


def project_2d(
    embeddings: np.ndarray, *, method: str = "auto", seed: int = RANDOM_SEED
) -> tuple[np.ndarray, str]:
    """Project the embeddings to 2-D for plotting (UMAP preferred, t-SNE fallback).

    Args:
        embeddings: ``(n, d)`` matrix.
        method: ``"umap"``, ``"tsne"``, or ``"auto"`` (UMAP if installed, else
            t-SNE).
        seed: Random seed for the projector (reproducible layout).

    Returns:
        A ``(coords, used_method)`` pair: an ``(n, 2)`` array and the name of the
        projector actually used.

    Raises:
        ValueError: If ``method`` is not ``"umap"``, ``"tsne"``, or ``"auto"``.
    """
    if method not in ("auto", "umap", "tsne"):
        raise ValueError(f"unknown method: {method!r}")

    if method in ("auto", "umap"):
        try:
            import umap

            logger.info("projecting %d vectors with UMAP", len(embeddings))
            reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed)
            return reducer.fit_transform(embeddings), "umap"
        except ImportError:
            if method == "umap":
                raise
            logger.info("umap-learn absent — falling back to t-SNE")

    from sklearn.manifold import TSNE

    logger.info("projecting %d vectors with t-SNE", len(embeddings))
    perplexity = min(TSNE_PERPLEXITY, max(5.0, (len(embeddings) - 1) / 3))
    tsne = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity, random_state=seed
    )
    return tsne.fit_transform(embeddings), "tsne"


def plot_projection(
    coords: np.ndarray,
    subjects: np.ndarray,
    out_path: Path,
    *,
    method: str = "",
    top_subjects: int = PLOT_TOP_SUBJECTS,
) -> Path | None:
    """Scatter the 2-D projection colored by subject; save to ``out_path``.

    The ``top_subjects`` most frequent subjects each get a color and a legend
    entry; everything else is drawn first in muted gray as a single "other"
    backdrop, so the highlighted subjects read clearly on top.

    Args:
        coords: ``(n, 2)`` projected points.
        subjects: Per-point subject string, row-aligned with ``coords``.
        out_path: Destination PNG (parent directories are created).
        method: Projector name, for the title (e.g. ``"umap"``).
        top_subjects: How many subjects to color individually.

    Returns:
        ``out_path`` on success, or ``None`` if matplotlib is not installed (the
        numbers still computed; only the figure is skipped — plan §1).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write a file, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib absent — skipping the 2-D map (install the 'viz' extra)"
        )
        return None

    subjects = np.asarray(subjects, dtype=object)
    top = pd.Series(subjects).value_counts().head(top_subjects).index.tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    other = ~np.isin(subjects, top)
    ax.scatter(
        coords[other, 0], coords[other, 1], s=3, c="lightgray", alpha=0.4, label="other"
    )
    cmap = plt.get_cmap("tab20")
    for i, subject in enumerate(top):
        m = subjects == subject
        ax.scatter(coords[m, 0], coords[m, 1], s=6, color=cmap(i % 20), label=subject)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Catalog embedding map ({method or '2-D'}) — top {top_subjects} subjects"
    )
    ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote 2-D map to %s", out_path)
    return out_path


def report_markdown(results: list[ClusterResult], *, model_name: str) -> str:
    """Render the per-algorithm coherence table as GitHub-flavored markdown."""
    cols = [
        "algorithm",
        "config",
        "n_clusters",
        "n_noise",
        "silhouette",
        "subject_purity",
        "largest_cluster_frac",
    ]
    header = (
        f"# Clustering diagnostic — {model_name}\n\n"
        "Diagnostic over the SBERT embeddings (plan §2.7 / Phase 6), not a "
        "leaderboard entry. `subject_purity` is the size-weighted dominant-subject "
        "share per cluster (text recovering subjects with no metadata); "
        "`silhouette` is the sampled cosine score. Regenerate with "
        "`python scripts/run_clustering.py`.\n\n"
    )
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in results:
        row = r.summary_row()
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return header + "\n".join(lines) + "\n"


def default_plot_path() -> Path:
    """Canonical destination for the Phase 6 embedding map."""
    return PLOTS_DIR / "embedding_map.png"
