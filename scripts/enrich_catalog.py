"""Populate the LLM tag cache via local Ollama (the slow, explicit Phase 7 pass).

Enrichment is split out from evaluation so ``scripts/run_eval.py`` stays fast and
deterministic: this script generates + caches tags once, and the recommender only
*reads* that cache. Resumable — re-run after an interruption and only the
remainder is generated.

By default it enriches the **eval-relevant subset** (cross-listing seeds + their
twins + judged-query gold courses) — the courses the leaderboard actually scores,
a few hundred, minutes on an M-series Mac. Pass ``--all`` for the full ~11k
catalog (a multi-hour one-time pass; needed eventually for the UI).

    python scripts/enrich_catalog.py                # eval-relevant subset
    python scripts/enrich_catalog.py --all          # whole catalog
    python scripts/enrich_catalog.py --model qwen3:32b

Requires a running Ollama daemon (`ollama serve`) with the model pulled; exits
with a clear message otherwise (the repo still runs end-to-end without it).
"""

from __future__ import annotations

import argparse
import logging

from courserec.config import DEFAULT_LLM_MODEL
from courserec.data import load_processed
from courserec.eval import build_crosslist_truth, load_judged_queries
from courserec.recommenders.llm import LLMUnavailable, OllamaClient, enrich_courses

logger = logging.getLogger(__name__)


def eval_relevant_ids(courses) -> list[str]:
    """Return the course ids the leaderboard scores against (sorted, de-duped).

    The union of every cross-listing seed and its twins (the primary + held-out
    lenses) and every judged-query gold course (the free-text lens) — enriching
    these makes both leaderboards meaningful at a fraction of the full-catalog
    cost.

    Args:
        courses: Processed catalog indexed by ``course_id``.

    Returns:
        The eval-relevant ids present in the catalog, in sorted order.
    """
    truth = build_crosslist_truth(courses)
    ids: set[str] = set()
    for seed, twins in truth.items():
        ids.add(seed)
        ids.update(twins)
    for q in load_judged_queries(catalog_ids=set(courses.index)):
        ids.update(q.relevant)
    return sorted(ids & set(courses.index))


def main() -> None:
    """Parse args, pick the id set, and enrich it via Ollama."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Enrich the whole catalog, not the subset."
    )
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL, help="Ollama model tag to enrich with."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    courses = load_processed()
    ids = list(courses.index) if args.all else eval_relevant_ids(courses)
    logger.info(
        "Enriching %d courses (%s) with %s",
        len(ids),
        "full catalog" if args.all else "eval-relevant subset",
        args.model,
    )

    client = OllamaClient(model=args.model)
    try:
        fresh = enrich_courses(courses, ids, client)
    except LLMUnavailable as exc:
        raise SystemExit(
            f"Ollama unavailable: {exc}\nStart it with `ollama serve` and pull the "
            f"model with `ollama pull {args.model}`."
        ) from exc
    logger.info("Done: %d freshly enriched of %d requested.", fresh, len(ids))


if __name__ == "__main__":
    main()
