"""Cached 2-D projection of the SBERT catalog for the Phase 8 Map view.

Projecting ~11k embeddings to 2-D is the slow step (t-SNE on the full catalog takes
tens of seconds; UMAP is faster), and the layout is deterministic given the seed — so
it is computed once and persisted to ``artifacts/map/`` rather than recomputed on
every interaction. This module owns that load-or-compute-and-cache logic and, like
:mod:`app.registry` / :mod:`app.glossary`, imports no Streamlit so it is unit-tested
on its own. The Streamlit layer wraps it in ``st.cache_resource`` for the in-session
cache and joins the coordinates with course metadata for plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from courserec.cluster import project_2d
from courserec.config import ARTIFACTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

#: Projected-coordinate caches live here (gitignored, like all of ``artifacts/``).
MAP_ARTIFACT_DIR: Path = ARTIFACTS_DIR / "map"


def projection_path(method: str, model_name: str) -> Path:
    """Return the cache path for a projection of ``model_name`` via ``method``.

    Args:
        method: Projector key (``"auto"``, ``"umap"``, or ``"tsne"``).
        model_name: SBERT model whose embeddings were projected (part of the key,
            so different embedders never collide).

    Returns:
        The ``.npy`` path under :data:`MAP_ARTIFACT_DIR`; the seed is in the name so a
        seed change can't silently serve a stale layout.
    """
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return MAP_ARTIFACT_DIR / f"coords_{method}_{safe_model}_seed{RANDOM_SEED}.npy"


def load_or_compute_coords(
    embeddings: np.ndarray,
    method: str,
    *,
    model_name: str,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, str]:
    """Load the cached 2-D projection, or compute it and cache it.

    Keyed by the *requested* ``method`` (not the one actually used), so repeated calls
    hit the same file even when ``"auto"`` resolves to UMAP or t-SNE under the hood.

    Args:
        embeddings: The ``(n, d)`` embedding matrix to project.
        method: ``"auto"`` (UMAP if installed, else t-SNE), ``"umap"``, or ``"tsne"``.
        model_name: SBERT model name, used in the cache key.
        seed: Random seed for a reproducible layout.

    Returns:
        A ``(coords, used_method)`` pair: an ``(n, 2)`` array and the projector name
        (the requested ``method`` on a cache hit; the one actually used on a miss).
    """
    path = projection_path(method, model_name)
    if path.exists():
        coords = np.load(path)
        if coords.shape == (len(embeddings), 2):
            logger.info("loaded cached projection from %s", path)
            return coords, method
        logger.warning(
            "stale projection cache %s (shape %s, expected (%d, 2)) — recomputing",
            path,
            coords.shape,
            len(embeddings),
        )
    coords, used = project_2d(embeddings, method=method, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coords)
    logger.info("cached %s projection to %s", used, path)
    return coords, used
