"""Fit every configured technique, score it, and write the leaderboard.

One command (``python scripts/run_eval.py``) regenerates
``results/leaderboard.{md,csv}`` — one row per technique×config, sorted by
NDCG@10, with bootstrap CIs on that primary metric. Never hand-edit the outputs.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from courserec.config import RESULTS_DIR
from courserec.data import load_processed
from courserec.eval import (
    EvalResult,
    build_crosslist_truth,
    build_reference_space,
    score_crosslist,
)
from courserec.interfaces import Recommender
from courserec.recommenders.lexical import BM25Recommender, TfidfRecommender

logger = logging.getLogger(__name__)


def build_recommenders() -> list[Recommender]:
    """Instantiate the Phase 1 lexical sweep (stopwords/n-grams/title weight)."""
    return [
        TfidfRecommender(ngram_max=1, title_weight=1),
        TfidfRecommender(ngram_max=2, title_weight=1),
        TfidfRecommender(ngram_max=1, title_weight=3),
        BM25Recommender(ngram_max=1, title_weight=1),
        BM25Recommender(ngram_max=1, title_weight=3),
    ]


def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no extra deps)."""
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_leaderboard(results: list[EvalResult]) -> pd.DataFrame:
    """Write ``results/leaderboard.{csv,md}`` sorted by NDCG@10 and return the frame.

    Args:
        results: One :class:`EvalResult` per technique×config.

    Returns:
        The sorted leaderboard as a DataFrame.
    """
    frame = pd.DataFrame([r.row() for r in results])
    frame = frame.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    header = (
        "# Leaderboard\n\n"
        "Sorted by NDCG@10. Regenerate with `python scripts/run_eval.py`.\n\n"
    )
    (RESULTS_DIR / "leaderboard.md").write_text(header + _to_markdown(frame))
    logger.info("Wrote leaderboard (%d rows) to %s", len(frame), RESULTS_DIR)
    return frame


def main() -> None:
    """Load data, fit + score every technique, and write the leaderboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for the NDCG@10 CI (use a small value to speed up).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    courses = load_processed()
    truth = build_crosslist_truth(courses)
    reference = build_reference_space(courses)

    results: list[EvalResult] = []
    for rec in build_recommenders():
        logger.info("Fitting %s", rec.name)
        start = time.perf_counter()
        rec.fit(courses)
        fit_time = time.perf_counter() - start
        results.append(
            score_crosslist(
                rec,
                courses,
                truth,
                reference,
                fit_time_s=fit_time,
                n_boot=args.bootstrap,
            )
        )

    frame = write_leaderboard(results)
    print(frame[["name", "ndcg@10", "ndcg@10_ci_low", "ndcg@10_ci_high", "recall@10"]])


if __name__ == "__main__":
    main()
