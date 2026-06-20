"""Generate "why this fits" explanations via local Ollama (Phase 7c / Track B.8c).

The explainer is a UI helper, not a ranked technique: given a seed course (or a
free-text query) it explains, in one sentence, why each top-k recommendation
fits. This script drives it from the command line — to validate the rung against
a live Ollama and to warm the explanation cache for the Phase 8 UI.

    python scripts/explain_recs.py --seed "AEROENG C124"
    python scripts/explain_recs.py --query "practical deep learning"
    python scripts/explain_recs.py --seed "AEROENG C124" --model qwen3:32b

Recommendations come from the SBERT MiniLM base (the top rung on both lenses).
Requires a running Ollama daemon (`ollama serve`) with the model pulled; prints a
clear message and exits otherwise (the repo still runs end-to-end without it).
"""

from __future__ import annotations

import argparse
import logging

from courserec.config import DEFAULT_LLM_MODEL
from courserec.data import load_processed
from courserec.recommenders.embeddings import SbertRecommender
from courserec.recommenders.llm import OllamaClient, RecommendationExplainer

logger = logging.getLogger(__name__)


def _print_explained(label: str, recs, explain) -> None:
    """Print each recommendation with its one-line justification (or a dash)."""
    print(f"\n{label}")  # noqa: T201 — a CLI tool's own stdout, not library logging
    for r in recs:
        reason = explain(r.course_id) or "—"
        print(f"  {r.course_id:<18} {r.score:.3f}  {reason}")  # noqa: T201


def main() -> None:
    """Parse args, retrieve top-k from SBERT, and explain each recommendation."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--seed", help="Seed course id for item-to-item recommendations."
    )
    group.add_argument("--query", help="Free-text query for search recommendations.")
    parser.add_argument("--k", type=int, default=5, help="How many recs to explain.")
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL, help="Ollama model tag to explain with."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    courses = load_processed()
    client = OllamaClient(model=args.model)
    if not client.available():
        raise SystemExit(
            f"Ollama unavailable at {client.host} ({args.model}).\nStart it with "
            f"`ollama serve` and pull the model with `ollama pull {args.model}`."
        )

    base = SbertRecommender()
    base.fit(courses)
    explainer = RecommendationExplainer(model=args.model).fit(courses)

    if args.seed:
        if args.seed not in courses.index:
            raise SystemExit(f"Unknown seed course id: {args.seed!r}")
        recs = base.recommend_similar(args.seed, k=args.k)
        _print_explained(
            f"Courses similar to {args.seed}:",
            recs,
            lambda cid: explainer.explain_seed(args.seed, cid),
        )
    else:
        recs = base.recommend_by_text(args.query, k=args.k)
        _print_explained(
            f"Courses for query {args.query!r}:",
            recs,
            lambda cid: explainer.explain(args.query, cid),
        )


if __name__ == "__main__":
    main()
