"""Run the Phase 6 clustering diagnostic and save the report + 2-D map.

One command (``python scripts/run_clustering.py``) clusters the cached SBERT
embeddings three ways, writes a coherence table to ``results/cluster_report.md``
(+ ``.csv``), and saves a subject-colored 2-D map to ``results/plots/``. It is a
diagnostic — it does not touch the leaderboard. With the ``semantic`` extra
missing it skips and flags rather than failing (plan §1).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from courserec.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_N_CLUSTERS,
    default_plot_path,
    load_sbert_embeddings,
    plot_projection,
    project_2d,
    report_markdown,
    run_clustering,
)
from courserec.config import RESULTS_DIR
from courserec.data import load_processed
from courserec.recommenders.embeddings import EmbeddingsUnavailable

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SBERT model id.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Cluster count for KMeans / agglomerative.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="HDBSCAN smallest admissible cluster.",
    )
    parser.add_argument(
        "--projection",
        choices=("auto", "umap", "tsne"),
        default="auto",
        help="2-D projector for the map (auto = UMAP if installed, else t-SNE).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the (slow) 2-D projection + figure; write only the table.",
    )
    return parser.parse_args()


def main() -> None:
    """Load embeddings, cluster, and write the report + 2-D map."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    courses = load_processed()
    try:
        embeddings, course_ids = load_sbert_embeddings(courses, model_name=args.model)
    except EmbeddingsUnavailable as exc:
        # Local-only guarantee: skip + flag, never fail the diagnostic.
        logger.warning("Clustering skipped — %s", exc)
        return

    subjects = courses.loc[course_ids, "subject"].to_numpy()
    results = run_clustering(
        embeddings,
        subjects,
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.summary_row() for r in results])
    frame.to_csv(RESULTS_DIR / "cluster_report.csv", index=False)
    (RESULTS_DIR / "cluster_report.md").write_text(
        report_markdown(results, model_name=args.model)
    )
    logger.info("Wrote cluster_report.{md,csv} to %s", RESULTS_DIR)
    print("\n== Clustering diagnostic ==")
    print(frame.to_string(index=False))

    if args.no_plot:
        logger.info("--no-plot set; skipping the 2-D map")
        return

    coords, method = project_2d(embeddings, method=args.projection)
    out = plot_projection(coords, subjects, default_plot_path(), method=method)
    if out is not None:
        print(f"\nSaved 2-D map ({method}) to {out}")


if __name__ == "__main__":
    main()
