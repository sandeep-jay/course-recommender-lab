# ADR-0008: Metadata fusion — weighted one-hot facets ⊕ TF-IDF (Phase 5 / B.5)

**Date:** 2026-06-19
**Status:** Accepted

## Context
Track B rung 5 (plan §2.5 / §5): "one/multi-hot of subject, department, level,
units; concatenate (weighted) with a text vector; tune weighting." Every prior
ranker compared courses on a single text signal; this one adds the catalog's
clean structured facets. Several decisions were load-bearing.

1. **Text backend: TF-IDF or SBERT?** The plan offers both. SBERT is the
   strongest text signal but is dense, needs the optional `semantic` extra, and
   mixing a 384-d dense block with a high-dimensional sparse one-hot block is
   awkward.
2. **How to fuse and how to parametrize the weighting.** "Weighted concatenate"
   leaves the normalization and the knob's meaning unspecified — and the knob is
   the one thing the plan says to tune.
3. **What "multi-hot" means for this catalog.** The plan says one/multi-hot, but
   the actual columns may be single-valued.
4. **How `recommend_by_text` should behave.** A free-text query has no subject or
   department — there is no facet block to build.
5. **Which lens validates it.** The primary automatic lens is cross-listing
   pairs; it is not obvious metadata helps there.

## Decision
1. **TF-IDF text backend, kept fully sparse.** The text block is a `TfidfVectorizer`
   matrix; the metadata block is a `DictVectorizer` one-hot. Both are sparse, so
   the fused matrix stays sparse and the technique needs **no extra and no key**
   (always runs, plan §1). Crucially this makes the metadata contribution
   **legible**: `metadata(text=1.0,…)` is bit-for-bit the existing
   `tfidf(sw=on,ng=1-1,tw=1)` baseline, so any delta is attributable to the
   fused facets and nothing else. Fusing onto SBERT is a noted extension, left
   out to keep the ablation clean.
2. **Per-block L2-normalize, scale by a single knob λ, concatenate.** Each block
   is unit-norm, so the inner product of two fused vectors is a weighted sum of
   two cosines: `score = λ²·cos_text + (1−λ)²·cos_meta`. `text_weight` (λ ∈ [0,1])
   is the one swept knob — λ=1 is pure text, λ=0 pure metadata. The squared
   coefficients are a documented consequence of scaling unit-norm blocks, not a
   second hyperparameter.
3. **Categorical one-hot; `units` bucketed by value.** Berkeley's `Department(s)`
   commas are part of names ("Theater, Dance, and Performance Studies"), not
   delimiters — subject/department/level are single-valued, so one-hot suffices
   (the `DictVectorizer` path stays multi-hot-capable for free). `units` reads
   `units_min` rendered as a compact label so all 3-unit courses share an
   indicator. Absent columns and null values are skipped, so a metadata-poor
   frame still fits (subject-only) rather than crashing.
4. **`recommend_by_text` zeroes the metadata block.** A query carries no facets,
   so its metadata block is zero and ranking reduces to weighted TF-IDF cosine —
   fusion helps only the item-to-item lens, never free text. `recommend_similar`
   reuses the seed's own fused row, so the seed's metadata *does* participate.
5. **Swept λ ∈ {0.9, 0.7, 0.5} on the cross-listing + judged-text leaderboards.**
   Joins the same harness as every other ranker; no new evaluation path.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|--------|------|------|-------------|
| SBERT text backend | Strongest text signal | Dense⊕sparse fusion is awkward; needs `semantic` extra + graceful skip; obscures the ablation | TF-IDF keeps it sparse, self-contained, and a clean delta vs the TF-IDF baseline |
| Per-facet weights (subject vs dept vs …) | Finer control | Many knobs, little signal at 11k rows; plan asks for one text↔metadata knob | One λ is the contract; per-facet tuning is unjustified complexity |
| Renormalize the full fused vector | "Tidy" unit-norm rows | Breaks the clean `λ²cos+ (1−λ)²cos` reading; shrinks text mass non-uniformly by how much metadata a doc has, distorting text queries | Per-block norm + no final renorm gives an interpretable, query-stable score |
| Bin `units` into hand-chosen bands | Fewer indicators | Bin edges are magic numbers; rare values (2.7, 3.3) are already few | Value-as-label needs no thresholds and the long tail is harmless |
| Treat dept commas as multi-value | Captures genuine cross-dept courses | The commas are inside proper names here — splitting fabricates departments | One-hot on the verbatim value is correct for this catalog |

## Consequences
**Positive:** A self-contained Track B ranker that always runs, with an exact
ablation hook (λ=1 ≡ the TF-IDF baseline) and a clean two-cosine score. The
sparse-text property is a genuine win: a description-less course has a near-empty
text block but a full metadata block, so fusion still places it sensibly where
text-only methods flounder.

**Negative / honest finding:** **metadata fusion loses on the primary lens, and
that is mechanistic, not a bug.** Cross-listing NDCG@10 falls monotonically as λ
drops — 0.948 (λ=0.9) → 0.936 (λ=0.7) → 0.909 (λ=0.5) — all *below* the plain
`tfidf` baseline at 0.955. The reason: **99.7% of cross-listing edges connect
different subjects and 97.0% different departments** (a cross-listing is, by
definition, the same course offered under two subject codes). So the one-hot
subject/department block actively *pushes a twin away* from its seed — fusion can
only hurt the lens that rewards twins ranking each other. On the judged free-text
lens all three λ tie *exactly* with the TF-IDF baseline (0.461), confirming the
zeroed-metadata design: text queries reduce to pure TF-IDF. The honest headline:
**these facets are orthogonal-to-adversarial against the cross-listing target;
their value is browse/filter coherence, which the harness only measures as a weak
same-subject proxy and explicitly warns not to optimize for.**

**Neutral:** λ is swept, not optimized — a higher-resolution sweep would not
change the direction (the gradient is monotone toward pure text). A same-subject
"browse" lens would show metadata in a favorable light, but the plan's three
lenses don't include one, and same-subject coherence is a known-degenerate
optimization target (rules/eval.md), so none was added.

## Implementation notes
`src/courserec/recommenders/metadata.py`: `MetadataRecommender` (`_build_docs`,
`_facet_dicts`/`_facet_value`, `_fuse`, fingerprint + artifact persistence shared
in spirit with the lexical rung). Sweep wired into `scripts/run_eval.py`
`build_recommenders`. Contract + fusion-behavior tests in `tests/test_metadata.py`
(λ=1 ≡ TF-IDF, metadata pulls same-facet up, sparse-text safe, missing-column
safe). Builds on the lexical rung ([ADR-0002](0002-eval-harness-design.md) eval
harness); the cross-listing-vs-metadata tension complements the graph rung's
held-out treatment of the same column ([ADR-0006](0006-graph-heldout.md)).
