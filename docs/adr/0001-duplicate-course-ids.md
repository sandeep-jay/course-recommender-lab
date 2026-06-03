# ADR-0001: Collapse duplicate course_ids to one representative row

**Date:** 2026-06-02
**Status:** Accepted

## Context
The plan (recommender_plan.md §0) specifies `course_id = f"{Subject} {Course
Number}"` and treats it as a stable, unique key — it is the recommendation
target, the index, and the key used to match cross-listing ground truth. In
practice the catalog violates this: 16 ids appear on 34 rows. Some are
byte-identical, others differ in description, title, units, or `cross_listed`.
A non-unique key would silently break index lookups, eval matching, and any
`recommend_similar` seed exclusion.

## Decision
Deduplicate on `course_id` in the loader, keeping one representative per id: the
row with the longest `text` field (most descriptive signal), ties broken by
first occurrence for determinism. Before dropping, coalesce `cross_listed`
across the group (forward/back-fill) so no ground-truth cross-listing edge is
lost when the kept row happens to be the one missing it. This reduces the
catalog from 11,091 raw rows to 11,073 unique courses.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| Keep duplicates, accept non-unique index | No data loss | Breaks index lookups, eval matching, seed exclusion; non-deterministic `.loc` | Unique key is load-bearing |
| Disambiguate ids (append a suffix) | Lossless | Invents ids that don't exist in the catalog; pollutes ground-truth matching | Cross-listings reference the real id |
| Drop only exact duplicates | Simple | Leaves 9 ids still colliding (differ in text/units) | Doesn't actually make the key unique |
| Keep first occurrence blindly | Trivial | Order-dependent; may keep the emptier row and lose a cross-listing | Determinism + ground-truth preservation matter |

## Consequences

**Positive:** `course_id` is a genuine unique key — index lookups, eval ground-truth
matching, and seed exclusion are all safe. Ground-truth edges are preserved via
coalescing. Deterministic (no seed needed).
**Negative:** 18 rows are discarded; a true distinct-section course sharing an id
(e.g. cross-campus `MBA 258` vs `EWMBA 258` variants) loses its alternate text.
This is acceptable at 0.16% of the catalog.
**Neutral:** Downstream row count is 11,073, not the plan's 11,091; tests and docs
assert the deduplicated figure.

## Implementation notes
`src/courserec/data.py::_deduplicate`, called from `load_raw` after `set_index`.
Coalesce uses `groupby(level=0)["cross_listed"].transform(ffill().bfill())`;
representative selection sorts by `text` length descending and keeps first.
`tests/test_data.py::test_course_ids_unique` and `test_row_count` lock the
invariant.
