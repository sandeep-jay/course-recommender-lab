#!/usr/bin/env python3
"""Validate and assemble the judged free-text query set (plan §3 lens 3).

The judged set (``data/judged_queries.json``) is hand-labeled ground truth: the
only way to evaluate ``recommend_by_text``. This helper does not invent labels —
curation stays human — but it makes that curation safe and cheap:

* ``--validate`` (default): every relevant ``course_id`` resolves against the
  current catalog, no query is empty, no duplicate labels. Exits non-zero on any
  problem so it can gate CI / a pre-commit hook.
* ``--suggest "<query>"``: print candidate ``course_id``s for a prospective
  query using the lexical TF-IDF recommender, to seed (never decide) hand labels.
* ``--stats``: summarize the set (query count, label counts, subject spread).

Usage:
    python scripts/build_judged_queries.py            # validate, print a summary
    python scripts/build_judged_queries.py --stats
    python scripts/build_judged_queries.py --suggest "practical deep learning"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter

from courserec.config import JUDGED_QUERIES_JSON
from courserec.data import load_processed
from courserec.eval import load_judged_queries
from courserec.recommenders.lexical import TfidfRecommender

logger = logging.getLogger(__name__)


def validate() -> list[str]:
    """Check the judged set against the catalog; return a list of problems.

    Returns:
        Human-readable problem strings (empty if the set is clean): unknown
        ``course_id`` references, queries with no relevant labels, duplicate
        queries, or duplicate labels within a query.
    """
    courses = load_processed()
    ids = set(courses.index)
    raw = json.loads(JUDGED_QUERIES_JSON.read_text())
    problems: list[str] = []

    seen_queries: set[str] = set()
    for entry in raw["queries"]:
        query = entry["query"]
        if query in seen_queries:
            problems.append(f"duplicate query: {query!r}")
        seen_queries.add(query)

        relevant = entry["relevant"]
        if len(relevant) != len(set(relevant)):
            problems.append(f"{query!r}: duplicate course_ids in relevant set")
        unknown = [c for c in relevant if c not in ids]
        if unknown:
            problems.append(f"{query!r}: unknown course_id(s) {unknown}")
        if not relevant:
            problems.append(f"{query!r}: empty relevant set")
    return problems


def stats() -> None:
    """Print a short summary of the judged set: counts and subject spread."""
    courses = load_processed()
    queries = load_judged_queries(catalog_ids=set(courses.index))
    label_counts = [len(q.relevant) for q in queries]
    subjects = Counter(
        courses.loc[cid, "subject"] for q in queries for cid in q.relevant
    )
    print(f"queries:        {len(queries)}")
    print(f"total labels:   {sum(label_counts)}")
    print(
        f"labels/query:   min={min(label_counts)} "
        f"max={max(label_counts)} mean={sum(label_counts) / len(label_counts):.1f}"
    )
    print(f"subjects hit:   {len(subjects)}")
    top = ", ".join(f"{s}×{n}" for s, n in subjects.most_common(8))
    print(f"top subjects:   {top}")


def suggest(query: str, k: int = 15) -> None:
    """Print lexical candidates for a prospective query, to seed hand labels.

    These are *suggestions only* — taking them verbatim would bias the ground
    truth toward lexical methods, defeating the lens. Read each candidate's title
    and keep only the genuinely on-topic ones.

    Args:
        query: The prospective natural-language query.
        k: How many candidates to print.
    """
    courses = load_processed()
    rec = TfidfRecommender(ngram_max=2)
    rec.fit(courses)
    print(f"Lexical candidates for {query!r} (curate by hand — do not paste verbatim):")
    for r in rec.recommend_by_text(query, k=k):
        title = str(courses.loc[r.course_id, "title"])
        print(f"  {r.score:.3f}  {r.course_id:15s}  {title[:60]}")


def main() -> None:
    """Parse arguments and run validate / stats / suggest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="Summarize the set.")
    parser.add_argument(
        "--suggest", metavar="QUERY", help="Print lexical candidates for a query."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.suggest:
        suggest(args.suggest)
        return
    if args.stats:
        stats()
        return

    problems = validate()
    if problems:
        print(f"INVALID — {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    print(f"valid ✓  ({JUDGED_QUERIES_JSON})")
    stats()


if __name__ == "__main__":
    main()
