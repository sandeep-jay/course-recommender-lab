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
    load_judged_queries,
    recommender_supports_text,
    score_crosslist,
    score_text_queries,
)
from courserec.interfaces import Recommender
from courserec.recommenders.embeddings import (
    ApiEmbeddingRecommender,
    EmbeddingsUnavailable,
    SbertRecommender,
)
from courserec.recommenders.lexical import BM25Recommender, TfidfRecommender
from courserec.recommenders.topics import (
    LDARecommender,
    LSARecommender,
    NMFRecommender,
)

logger = logging.getLogger(__name__)


def build_recommenders() -> list[Recommender]:
    """Instantiate the leaderboard sweep across the Phase 1–3 techniques.

    Phase 1 lexical rung (stopwords / n-grams / title weight), the Phase 2
    latent-topic rung (LSA / NMF / LDA), and the Phase 3 semantic rung (a small
    and a larger SBERT, plus an API embedding model). Every technique conforms to
    the same interface, so each new entry simply grows the leaderboard. The
    semantic entries need the optional ``semantic`` extra and (for the API one) a
    key; they skip gracefully when absent.
    """
    return [
        # Phase 1 — lexical baselines.
        TfidfRecommender(ngram_max=1, title_weight=1),
        TfidfRecommender(ngram_max=2, title_weight=1),
        TfidfRecommender(ngram_max=1, title_weight=3),
        BM25Recommender(ngram_max=1, title_weight=1),
        BM25Recommender(ngram_max=1, title_weight=3),
        # Phase 2 — latent topics. LSA tolerates many components; NMF and LDA
        # favor a smaller, interpretable topic count.
        LSARecommender(n_topics=200, ngram_max=1, title_weight=1),
        NMFRecommender(n_topics=50, ngram_max=1, title_weight=1),
        LDARecommender(n_topics=50, ngram_max=1, title_weight=1),
        # Phase 3 — semantic vectors. Small + larger local SBERT, then a hosted
        # API embedding model (skipped + flagged when no key is set).
        SbertRecommender(model_name="all-MiniLM-L6-v2"),
        SbertRecommender(model_name="all-mpnet-base-v2"),
        ApiEmbeddingRecommender(model_name="text-embedding-3-small"),
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


def write_leaderboard(
    results: list[EvalResult],
    *,
    stem: str,
    title: str,
    subtitle: str,
    drop_cols: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Write ``results/<stem>.{csv,md}`` sorted by NDCG@10 and return the frame.

    Args:
        results: One :class:`EvalResult` per technique×config for this lens.
        stem: Output filename stem (``leaderboard`` or ``leaderboard_text``).
        title: Markdown H1 for the ``.md`` file.
        subtitle: One-line description under the title.
        drop_cols: Columns to omit (e.g. ``same_subject@10`` for the text lens,
            where it is undefined).
        notes: Blockquote lines flagging any skipped/unavailable techniques, so a
            gap is recorded in the output, never silently omitted (plan §3).

    Returns:
        The sorted leaderboard as a DataFrame.
    """
    frame = pd.DataFrame([r.row() for r in results])
    frame = frame.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    if drop_cols:
        frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / f"{stem}.csv", index=False)
    header = f"# {title}\n\n{subtitle}\n\n"
    for note in notes:
        header += f"> {note}\n\n"
    (RESULTS_DIR / f"{stem}.md").write_text(header + _to_markdown(frame))
    logger.info("Wrote %s (%d rows) to %s", stem, len(frame), RESULTS_DIR)
    return frame


def _skip_note(label: str, names: list[str], reason: str) -> tuple[str, ...]:
    """Build a one-line blockquote note for skipped techniques (empty if none)."""
    if not names:
        return ()
    listed = ", ".join(f"`{n}`" for n in names)
    return (f"**{label} ({len(names)}):** {listed} — {reason}.",)


def main() -> None:
    """Load data, fit + score every technique on both lenses, write leaderboards."""
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
    judged = load_judged_queries(catalog_ids=set(courses.index))

    crosslist: list[EvalResult] = []
    text: list[EvalResult] = []
    text_skipped: list[str] = []
    unavailable: list[str] = []
    for rec in build_recommenders():
        logger.info("Fitting %s", rec.name)
        start = time.perf_counter()
        try:
            rec.fit(courses)
        except EmbeddingsUnavailable as exc:
            # API/optional-dep techniques degrade gracefully: skip + flag, never
            # fail the suite (the repo must run local-only with no key).
            logger.warning("%s: unavailable — %s", rec.name, exc)
            unavailable.append(rec.name)
            continue
        fit_time = time.perf_counter() - start
        crosslist.append(
            score_crosslist(
                rec,
                courses,
                truth,
                reference,
                fit_time_s=fit_time,
                n_boot=args.bootstrap,
            )
        )
        # Free-text lens — skip (and flag) techniques that cannot serve text.
        if recommender_supports_text(rec):
            text.append(
                score_text_queries(
                    rec,
                    judged,
                    courses,
                    reference,
                    fit_time_s=fit_time,
                    n_boot=args.bootstrap,
                )
            )
        else:
            text_skipped.append(rec.name)
            logger.warning(
                "%s: no recommend_by_text — skipped on the text lens", rec.name
            )

    unavailable_note = _skip_note(
        "Unavailable", unavailable, "missing optional dependency or API key"
    )
    frame = write_leaderboard(
        crosslist,
        stem="leaderboard",
        title="Leaderboard — cross-listing lens",
        subtitle="Sorted by NDCG@10. Regenerate with `python scripts/run_eval.py`.",
        notes=unavailable_note,
    )
    print("\n== Cross-listing lens ==")
    print(frame[["name", "ndcg@10", "ndcg@10_ci_low", "ndcg@10_ci_high", "recall@10"]])

    if text:
        text_frame = write_leaderboard(
            text,
            stem="leaderboard_text",
            title="Leaderboard — judged free-text lens",
            subtitle=(
                f"`recommend_by_text` over {text[0].n_queries} hand-labeled queries "
                "(plan §3 lens 3). Sorted by NDCG@10."
            ),
            drop_cols=("same_subject@10",),
            notes=unavailable_note
            + _skip_note(
                "Text mode not implemented",
                text_skipped,
                "technique is item-to-item only",
            ),
        )
        print("\n== Judged free-text lens ==")
        print(
            text_frame[
                ["name", "ndcg@10", "ndcg@10_ci_low", "ndcg@10_ci_high", "recall@10"]
            ]
        )


if __name__ == "__main__":
    main()
