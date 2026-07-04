# The Data — UC Berkeley course catalog

**Everything in this lab is recommended over one dataset: the UC Berkeley course
catalog.** No clicks, no ratings, no enrollments — just the text and metadata the
university publishes for each course. That single constraint shapes the whole project,
so it's worth seeing exactly what the raw material is before you read a single result.

<div class="grid cards" markdown>

- :material-database-outline: **11,073 courses** after cleaning
- :material-tag-multiple-outline: **242 subjects** across **112 departments**
- :material-link-variant: **1,080 cross-listed** (9.8%) — the evaluation ground truth
- :material-text-box-outline: **~2% sparse text** — 226 missing + 229 one-word descriptions

</div>

## Where it comes from

The source is Berkeley's public undergraduate course catalog:

> **[undergraduate.catalog.berkeley.edu/courses](https://undergraduate.catalog.berkeley.edu/courses?cq=&sortBy=code&page=1)**

The extract used here is a one-time CSV export
(`data/raw/courses-report_2026-06-02.csv`, 14 columns, ~11k rows). It is a **static
snapshot** — the lab never scrapes the live site, so results are reproducible against a
fixed file. There is deliberately **no user-interaction data**: nobody's click history,
no ratings. That is why [collaborative filtering is out of scope](reviewer-guide.md) and
every technique here is **content-based** — it must earn its ranking from course text
and metadata alone.

## What one course looks like

A single real record — Berkeley's introductory data-science course — carried through the
cleaning pipeline:

| Field | Value |
|---|---|
| `course_id` | `DATA C8` *(synthesized as `f"{subject} {course_number}"`)* |
| `subject` | `DATA` |
| `department` | Data Science Undergraduate Studies |
| `title` | Foundations of Data Science |
| `level` | `lower-div` *(parsed from the course number)* |
| `units_min` – `units_max` | 4.0 – 4.0 |
| `description` | *"Foundations of data science from three perspectives: inferential thinking, computational thinking, and real-world relevance…"* |
| `cross_listed` | `STATC8…, INFOC8…, COMPSCIC8…` *(the same course under three other subjects)* |
| `text` | `"Foundations of Data Science. Foundations of data science from three perspectives…"` *(title + description — the signal every technique consumes)* |

## The schema

The export has 14 raw columns; the loader keeps **8** and drops the rest (Terms Offered,
Repeat Rules, Offering Details, and other boilerplate that carries no recommendation
signal). The kept fields:

| Processed field | Source column | Role |
|---|---|---|
| `course_id` | *synthesized* | Unique key — the recommendation and match key |
| `subject` | Subject | Metadata facet |
| `course_number` | Course Number | Feeds `course_id` and `level` |
| `department` | Department(s) | Metadata facet |
| `title` | Course Title | Text signal (100% populated) |
| `description` | Course Description | Text signal (may be sparse) |
| `level` | *parsed* | `lower-div` / `upper-div` / `grad`, from the numeric core |
| `units_min`, `units_max` | Credits – Units | Numeric metadata (0–15) |
| `cross_listed` | Cross-Listed Course(s) | :material-alert: **Ground truth — never an input feature** |

`text = "{title}. {description}"` is the combined field every technique reads. When a
description is missing or a single word, it **falls back to the title** — so no technique
ever crashes on empty text.

## The ground truth: cross-listings

This is the single most important thing to understand about the dataset, because the
whole evaluation hinges on it.

A **cross-listed course** is *one* course that appears under several subject codes.
"Foundations of Data Science" is literally the same class whether you find it as
**DATA C8**, **STAT C8**, **COMPSCI C8**, or **INFO C8** — same instructor, same
description, one course, four listings. Those listings are **twins**.

!!! success "Why twins make a free ground-truth signal"
    Nobody hand-labeled "these two courses are similar." The catalog already tells us:
    if a good recommender is shown DATA C8, the STAT/COMPSCI/INFO twins *should* rank at
    the very top. That gives us an automatic relevance label for **1,080 courses** — no
    annotation budget required. It is the primary lens the [leaderboard](RESULTS.md) is
    scored on.

!!! danger "The leakage rule — the one bug that would invalidate everything"
    Because `cross_listed` **is** the answer key, **no technique may read it as an input
    feature.** A model that peeked at the cross-listing column would "predict" twins
    perfectly and score a meaningless 1.0. Every recommender is built blind to that
    column; the one graph technique that *does* use cross-listing edges is evaluated
    only on a **held-out split** it never saw during fitting
    ([ADR-0006](adr/0006-graph-heldout.md)).

Two properties of the twins are worth carrying into the results:

- **Twins are near-identical in text** — often the exact same description. That makes the
  cross-listing lens *easy* for even simple lexical methods, so it validates
  **correctness** more than quality. We report bootstrap confidence intervals and never
  crown a winner on a gap smaller than the CIs ([Architecture](ARCHITECTURE.md#the-evaluation-harness)).
- **Twins usually span subjects** — that's the whole point of a cross-listing. So
  `subject`/`department` metadata actively *misleads* on this target, which is exactly
  why [metadata fusion hurts](adr/0008-metadata-fusion.md) rather than helps.

## The quirks the loader handles

Real catalog data is messy. The cleaning pipeline
([`src/courserec/data.py`](https://github.com/sandeep-jay/course-recommender-lab/blob/main/src/courserec/data.py))
handles four traps that would silently corrupt results if ignored:

| Trap | What breaks if you ignore it | How it's handled |
|---|---|---|
| **The `"-"` null token** | The catalog writes a literal `"-"` for "no value". Treated as text, every empty field becomes a fake shared token. | Replaced with real `NA` on load, before anything reads it. |
| **Quoted newlines** | Descriptions contain line breaks *inside* quoted CSV fields. A line-based reader shreds them into broken half-rows. | Parsed with pandas (RFC-4180 aware), never line-by-line. |
| **Sparse descriptions** | ~2% of courses have a missing or one-word description. Vectorizing empty text yields a zero vector or a crash. | Falls back to the title; never crashes on empty text. |
| **Duplicate course ids** | A handful of courses collide on `course_id`, which must be unique to serve as the match key. | Collapsed to the richest row, coalescing cross-listing edges ([ADR-0001](adr/0001-duplicate-course-ids.md)). |

## What's in the catalog

<div class="grid cards" markdown>

- **By level**
    - upper-division — 5,003
    - graduate — 4,349
    - lower-division — 1,721

- **By breadth**
    - 242 subjects (e.g. `COMPSCI`, `STAT`, `HISTORY`, `MUSIC`)
    - 112 departments
    - units range 0–15

</div>

---

**Next:** see how these 11,073 courses are turned into rankings and scored in the
[Architecture](ARCHITECTURE.md), or watch a technique built from primitives on this exact
data in [Notebook 00 — Data & Eval foundation](notebooks/00_data_and_eval.ipynb).
