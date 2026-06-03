# ADR-0002: Cross-listing ground truth and a technique-agnostic diversity space

**Date:** 2026-06-02
**Status:** Accepted

## Context
Phase 1 needs the shared evaluation harness (`src/courserec/eval.py`). Two design
points were non-obvious and load-bearing for every future technique:

1. **Resolving cross-listing ground truth.** The catalog's `Cross-Listed
   Course(s)` cell is the primary relevance signal, but it is not a clean id. It
   stores `"{SUBJECT}{NUMBER} {truncated title}"` with the subject and number
   space-stripped (e.g. `MATSCIC135 MATS EXTRM ENVRMTS`), comma-separated for
   multiples — whereas the catalog key is `f"{Subject} {Course Number}"`
   (`MATSCI C135`). Subjects vary in length and contain no internal delimiter, so
   splitting `MATSCIC135` into subject + number directly is ambiguous.
2. **Measuring intra-list diversity.** Diversity needs a notion of item-item
   similarity. The obvious choice — the candidate model's own vector space —
   flatters that model and makes diversity incomparable across techniques.

## Decision
1. **Resolve by space-stripped lookup.** Build `{course_id.replace(" ",""):
   course_id}` once, then match each reference's leading whitespace-delimited
   token against it. References to courses outside the catalog (and self-refs)
   are silently dropped. This resolves ~96% of references (1,454/1,513); the
   unresolved remainder are genuinely not in the catalog and cannot be ground
   truth anyway. Yields 1,072 seeds with ≥1 in-catalog twin.
2. **Diversity in a fixed, technique-agnostic reference space.** Fit one TF-IDF
   space over the combined text once (`build_reference_space`) and measure
   intra-list diversity as mean pairwise cosine *distance* there, identically for
   every technique.

Supporting choices, for the record: same-subject coherence is reported as a
sanity floor and never optimized; the primary metric (NDCG@10) ships with a
percentile bootstrap CI under the global seed, because the truth set is small.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Parse subject/number from the token with a regex | No prebuilt lookup | Subject length is ambiguous (no delimiter); fragile across 242 subjects | Space-stripped lookup is exact and trivial |
| Treat the whole cell (incl. truncated title) as the match key | Uses more of the string | Titles are truncated/inconsistent; brittle | Leading token alone is sufficient and stable |
| Measure diversity in the model's own space | "Native" to the technique | Flatters the model; not comparable across techniques | Diversity must be a neutral yardstick |
| Skip diversity/novelty, report only relevance | Less code | Lets a model "win" the trivial twin lens while recommending redundant lists | Plan §6 mandates these guardrails |

## Consequences
**Positive:** Ground truth is exact and reproducible; diversity/novelty/coverage
are comparable across every present and future technique on one leaderboard.
Bootstrap CIs prevent crowning a winner on noise.
**Negative:** ~4% of cross-listing references are unresolved (out-of-catalog);
the reference TF-IDF space is itself a lexical lens, so diversity is measured
through a bag-of-words lens even for semantic models — acceptable as a fixed,
neutral ruler, but noted.
**Neutral:** The cross-listing lens is near-trivial for lexical methods (twins
share text); this is expected and is why the judged-query set (plan §3 lens 3)
remains an open gap.

## Implementation notes
`src/courserec/eval.py`: `_resolve_refs` / `build_crosslist_truth` (resolution),
`build_reference_space` / `intra_list_diversity` (diversity), `bootstrap_ci`
(CIs). Tests in `tests/test_eval.py` lock resolution, hand-checked metric values,
and bootstrap determinism.
