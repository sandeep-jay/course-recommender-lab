"""Tests for the Phase 8 Map projection cache (`app/projection.py`).

The projection is the slow step (t-SNE/UMAP over ~11k vectors), so the module's job
is to compute it at most once and reuse it. These tests pin that contract — a fresh
key computes and writes the cache, a matching key reads it back without recomputing,
and a shape mismatch recomputes rather than serving a stale layout — using a stub
projector so they stay fast and never import Streamlit or run a real reducer.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import projection


@pytest.fixture
def patched_dir(tmp_path, monkeypatch):
    """Point the projection cache at a temp dir for the duration of a test."""
    monkeypatch.setattr(projection, "MAP_ARTIFACT_DIR", tmp_path / "map")
    return tmp_path / "map"


def _stub_project(coords, used="tsne"):
    """Build a project_2d stand-in that records its call count."""
    calls = {"n": 0}

    def _fake(embeddings, *, method, seed):
        calls["n"] += 1
        return coords, used

    return _fake, calls


def test_projection_path_encodes_method_model_seed(patched_dir):
    """The cache filename carries method, sanitized model, and the seed."""
    path = projection.projection_path("tsne", "all-MiniLM-L6-v2")
    assert path.parent == patched_dir
    assert path.name == f"coords_tsne_all_MiniLM_L6_v2_seed{projection.RANDOM_SEED}.npy"


def test_computes_and_caches_on_miss(patched_dir, monkeypatch):
    """A cold key computes via project_2d, writes the file, returns its method."""
    embeddings = np.zeros((5, 3), dtype="float32")
    coords = np.arange(10, dtype="float32").reshape(5, 2)
    fake, calls = _stub_project(coords, used="umap")
    monkeypatch.setattr(projection, "project_2d", fake)

    out, used = projection.load_or_compute_coords(
        embeddings, "auto", model_name="all-MiniLM-L6-v2"
    )

    assert calls["n"] == 1
    assert used == "umap"
    np.testing.assert_array_equal(out, coords)
    assert projection.projection_path("auto", "all-MiniLM-L6-v2").exists()


def test_reuses_cache_on_hit(patched_dir, monkeypatch):
    """A warm key of matching shape returns the cached coords without recomputing."""
    embeddings = np.zeros((5, 3), dtype="float32")
    coords = np.arange(10, dtype="float32").reshape(5, 2)
    path = projection.projection_path("tsne", "all-MiniLM-L6-v2")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)

    def _boom(*a, **k):
        raise AssertionError("project_2d must not run on a cache hit")

    monkeypatch.setattr(projection, "project_2d", _boom)

    out, used = projection.load_or_compute_coords(
        embeddings, "tsne", model_name="all-MiniLM-L6-v2"
    )
    assert used == "tsne"
    np.testing.assert_array_equal(out, coords)


def test_stale_shape_recomputes(patched_dir, monkeypatch):
    """A cached layout whose row count no longer matches is recomputed, not served."""
    embeddings = np.zeros((5, 3), dtype="float32")
    stale = np.zeros((4, 2), dtype="float32")  # wrong row count
    path = projection.projection_path("tsne", "all-MiniLM-L6-v2")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stale)

    fresh = np.ones((5, 2), dtype="float32")
    fake, calls = _stub_project(fresh, used="tsne")
    monkeypatch.setattr(projection, "project_2d", fake)

    out, _ = projection.load_or_compute_coords(
        embeddings, "tsne", model_name="all-MiniLM-L6-v2"
    )
    assert calls["n"] == 1
    np.testing.assert_array_equal(out, fresh)
