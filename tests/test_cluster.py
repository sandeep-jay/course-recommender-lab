"""Tests for the Phase 6 clustering diagnostic (src/courserec/cluster.py).

The diagnostic operates on plain embedding matrices, so these tests use small
synthetic well-separated blobs — no SBERT, no network, fast. They assert the
contract: labels align with the input, coherence metrics behave on a known-good
partition, the noise label is handled, and the 2-D map + report render.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from courserec.cluster import (
    ClusterResult,
    _silhouette,
    _subject_purity,
    cluster_embeddings,
    evaluate_clustering,
    plot_projection,
    project_2d,
    report_markdown,
    run_clustering,
)
from courserec.config import RANDOM_SEED


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated unit-norm blobs with a subject per blob.

    Returns an ``(embeddings, subjects)`` pair: 90 points (30 per blob) in 8-D,
    L2-normalized, where the true subject is the blob id — so a sane clustering
    into three groups should recover near-perfect subject purity.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    centers = np.eye(3, 8)  # three orthogonal directions in 8-D
    pts, subjects = [], []
    for i, center in enumerate(centers):
        cloud = center + 0.02 * rng.standard_normal((30, 8))
        pts.append(cloud)
        subjects += [f"SUBJ{i}"] * 30
    emb = np.vstack(pts).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, np.array(subjects, dtype=object)


@pytest.mark.parametrize("algorithm", ["kmeans", "agglomerative", "hdbscan"])
def test_cluster_embeddings_labels_align(blobs, algorithm):
    """Every algorithm returns one integer label per row, plus its config dict."""
    emb, _ = blobs
    labels, config = cluster_embeddings(
        emb, algorithm=algorithm, n_clusters=3, min_cluster_size=5
    )
    assert labels.shape == (len(emb),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert isinstance(config, dict) and config


def test_cluster_embeddings_rejects_unknown_algorithm(blobs):
    """An unsupported algorithm name is a programming error, not silent garbage."""
    emb, _ = blobs
    with pytest.raises(ValueError, match="unknown algorithm"):
        cluster_embeddings(emb, algorithm="dbscan")


def test_kmeans_recovers_blobs(blobs):
    """KMeans at the true k finds 3 clusters with perfect subject purity."""
    emb, subjects = blobs
    labels, config = cluster_embeddings(emb, algorithm="kmeans", n_clusters=3)
    result = evaluate_clustering(
        emb, labels, subjects, algorithm="kmeans", config=config
    )
    assert result.n_clusters == 3
    assert result.n_noise == 0
    assert result.subject_purity == pytest.approx(1.0)
    assert result.silhouette > 0.5  # well-separated blobs are tightly cohesive


def test_subject_purity_perfect_and_mixed():
    """Purity is 1.0 when clusters are single-subject, lower when mixed."""
    subjects = np.array(["A", "A", "B", "B"], dtype=object)
    assert _subject_purity(np.array([0, 0, 1, 1]), subjects) == pytest.approx(1.0)
    # one cluster holding both subjects -> dominant share is 2/4.
    assert _subject_purity(np.array([0, 0, 0, 0]), subjects) == pytest.approx(0.5)


def test_subject_purity_ignores_noise():
    """Noise points (-1) are excluded from the purity denominator."""
    subjects = np.array(["A", "A", "B"], dtype=object)
    # only the two clustered points count; both are subject A -> purity 1.0.
    assert _subject_purity(np.array([0, 0, -1]), subjects) == pytest.approx(1.0)
    assert _subject_purity(np.array([-1, -1, -1]), subjects) == 0.0


def test_silhouette_nan_with_single_cluster(blobs):
    """Silhouette is undefined (nan) when there are fewer than two clusters."""
    emb, _ = blobs
    assert math.isnan(_silhouette(emb, np.zeros(len(emb), dtype=int)))


def test_evaluate_clustering_handles_noise(blobs):
    """A labeling with noise reports n_noise and excludes it from cluster counts."""
    emb, subjects = blobs
    labels = np.zeros(len(emb), dtype=int)
    labels[:5] = -1  # mark five points as noise
    labels[5:45] = 1
    result = evaluate_clustering(
        emb, labels, subjects, algorithm="hdbscan", config={"min_cluster_size": 5}
    )
    assert result.n_noise == 5
    assert result.n_clusters == 2  # labels {0, 1}; -1 is not a cluster
    assert 0.0 < result.largest_cluster_frac <= 1.0


def test_run_clustering_one_result_per_algorithm(blobs):
    """run_clustering returns a populated ClusterResult per requested algorithm."""
    emb, subjects = blobs
    results = run_clustering(
        emb, subjects, algorithms=("kmeans", "agglomerative"), n_clusters=3
    )
    assert [r.algorithm for r in results] == ["kmeans", "agglomerative"]
    assert all(isinstance(r, ClusterResult) for r in results)


def test_project_2d_tsne_shape_and_unknown_method(blobs):
    """t-SNE returns (n, 2) coords; an unknown method is rejected."""
    emb, _ = blobs
    coords, method = project_2d(emb, method="tsne")
    assert coords.shape == (len(emb), 2)
    assert method == "tsne"
    with pytest.raises(ValueError, match="unknown method"):
        project_2d(emb, method="pca")


def test_plot_projection_writes_file(blobs, tmp_path):
    """The 2-D map renders to a PNG (matplotlib is in the 'viz' extra)."""
    pytest.importorskip("matplotlib")
    emb, subjects = blobs
    coords, _ = project_2d(emb, method="tsne")
    out = plot_projection(coords, subjects, tmp_path / "map.png", method="tsne")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_report_markdown_lists_algorithms(blobs):
    """The report table names each algorithm and the purity column."""
    emb, subjects = blobs
    results = run_clustering(emb, subjects, algorithms=("kmeans",), n_clusters=3)
    md = report_markdown(results, model_name="all-MiniLM-L6-v2")
    assert "kmeans" in md
    assert "subject_purity" in md
    assert "all-MiniLM-L6-v2" in md
