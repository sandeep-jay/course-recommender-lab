"""Plain-language descriptions for techniques, metrics, and eval lenses.

The UI shows cryptic technique keys (`sbert(all_minilm_l6_v2,idx=flat)`) and metric
columns (`ndcg@10`, `mrr`, `novelty`) with no hint of what they mean. This module is
the explanatory layer: it maps each exposed technique to a one-line blurb, each
technique *family* to a paragraph (so even leaderboard rows the UI never fits are
explained), each leaderboard metric to a definition, and the three evaluation lenses
to their meaning and caveats.

Like :mod:`app.registry` it imports no Streamlit, so the content is unit-tested in
the base environment; :mod:`app.streamlit_app` renders it (picker captions, column
tooltips, leaderboard expanders). Descriptions stay honest to the project's findings
(RESULTS.md / the ADRs) — e.g. metadata fusion *hurting* the cross-listing target is
stated, not hidden.
"""

from __future__ import annotations

# --- techniques: per-rung blurbs for the picker (keyed by UI display name) ---
# Keys must match app.registry.TECHNIQUE_FACTORIES (a test enforces this).
TECHNIQUE_INFO: dict[str, str] = {
    "SBERT MiniLM (top rung)": (
        "Semantic embeddings — a 384-dim MiniLM bi-encoder maps each course's text "
        "to a vector; similarity is cosine distance. The project's strongest "
        "technique: tops both ranking lenses and wins free-text decisively. Default."
    ),
    "SBERT MPNet": (
        "Semantic embeddings — a larger 768-dim MPNet bi-encoder. Essentially tied "
        "with MiniLM on these lenses, at higher fit/query cost."
    ),
    "TF-IDF (unigram)": (
        "Classic lexical baseline — TF-IDF term vectors compared by cosine. Nails "
        "near-identical cross-listed twins; weaker on paraphrased free-text queries."
    ),
    "BM25": (
        "Okapi BM25 — TF-IDF's cousin with saturating term frequency and document-"
        "length normalization. Strong on twin text, lexical-literal like TF-IDF."
    ),
    "LSA (200 topics)": (
        "Latent Semantic Analysis — truncated SVD to 200 latent topics over the "
        "TF-IDF matrix. Captures synonymy but blurs fine-grained detail."
    ),
    "Metadata + text": (
        "Blends the text vector with weighted subject/department/level/units facets. "
        "Note: those facets *hurt* the cross-listing target here, because ~99.7% of "
        "twins span subjects (ADR-0008)."
    ),
}

# --- families: a paragraph per family, so any leaderboard row is explainable ---
# Human label shown in the leaderboard's "family" column.
FAMILIES: dict[str, str] = {
    "semantic": "Semantic embeddings",
    "lexical": "Lexical (bag-of-words)",
    "topic": "Topic model",
    "metadata": "Metadata fusion",
    "rerank": "Cross-encoder rerank",
    "llm_tags": "LLM tag profiles",
    "llm_rerank": "LLM reranker",
    "graph": "Course graph",
    "other": "Other",
}

# Map a raw technique-name prefix (everything before the first "(") to a family key.
_FAMILY_BY_PREFIX: dict[str, str] = {
    "sbert": "semantic",
    "api_embed": "semantic",
    "tfidf": "lexical",
    "bm25": "lexical",
    "lsa": "topic",
    "nmf": "topic",
    "lda": "topic",
    "metadata": "metadata",
    "rerank": "rerank",
    "llm_tags": "llm_tags",
    "llm_rerank": "llm_rerank",
    "graph": "graph",
}

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "semantic": (
        "Sentence-BERT bi-encoders embed each course's text into one dense vector; "
        "similarity is cosine distance in that space. The strongest family here — "
        "MiniLM tops both ranking lenses and beats every lexical config on free text."
    ),
    "lexical": (
        "Bag-of-words scoring over shared terms — TF-IDF cosine, or Okapi BM25's "
        "saturating term frequency. Strong on cross-listing (twins share near-"
        "identical wording) but weaker when a query paraphrases the catalog."
    ),
    "topic": (
        "Reduce the term–document matrix to a few latent topics (LSA = truncated "
        "SVD, NMF = non-negative parts, LDA = probabilistic topics) and compare "
        "topic vectors. Smooths vocabulary at the cost of fine-grained detail."
    ),
    "metadata": (
        "Blends the text vector with weighted one-hot facets (subject, department, "
        "level, units). It *hurts* the cross-listing target because ~99.7% of twins "
        "span subjects, so subject facets push true twins apart (ADR-0008)."
    ),
    "rerank": (
        "Two stages: SBERT retrieves a candidate pool, a cross-encoder re-scores "
        "each query–candidate pair jointly, then MMR trades relevance for diversity "
        "(λ). The SBERT pool is already near-ceiling, so it adds latency, not lift."
    ),
    "llm_tags": (
        "A local LLM (qwen3:8b) distills each course to ~6–12 tags; ranking is "
        "TF-IDF cosine over tag profiles. Full-catalog enrichment overturned an "
        "early win — distillation loses more signal than it adds (ADR-0009)."
    ),
    "llm_rerank": (
        "SBERT retrieves the top-20, then a local LLM reorders them over their full "
        "text in one deterministic call. Measured: does not beat the SBERT base, at "
        "~4 s/query (ADR-0010)."
    ),
    "graph": (
        "A course graph with personalized PageRank, evaluated only on a held-out "
        "split of cross-listing edges (the one technique allowed to use that column)."
    ),
    "other": "Technique family not recognized — see the technique name and config.",
}

# --- metrics: leaderboard column -> plain-English definition --------------------
METRIC_GLOSSARY: dict[str, str] = {
    "n_queries": "Number of evaluation queries scored (seeds with a known twin).",
    "recall@5": "Share of true relevant courses found in the top 5. Higher is better.",
    "recall@10": "Share of true relevant courses found in the top 10.",
    "recall@20": "Share of true relevant courses found in the top 20.",
    "precision@5": (
        "Share of the top 5 that are truly relevant. With ~1 twin per seed this is "
        "mechanically low — judge recall/NDCG, not precision, on this lens."
    ),
    "precision@10": "Share of the top 10 that are truly relevant (mechanically low).",
    "precision@20": "Share of the top 20 that are truly relevant (mechanically low).",
    "ndcg@5": "NDCG at 5 — rewards ranking relevant courses higher, scaled to [0,1].",
    "ndcg@10": (
        "Normalized Discounted Cumulative Gain at 10 — the primary metric and the "
        "default sort. Rewards putting relevant courses near the top; 1.0 is perfect."
    ),
    "ndcg@20": "NDCG at 20 — the same measure over a top-20 list.",
    "map": "Mean Average Precision — precision averaged across recall levels.",
    "mrr": (
        "Mean Reciprocal Rank — 1/rank of the first relevant hit, averaged. 1.0 "
        "means the twin is always ranked first."
    ),
    "ndcg@10_ci_low": (
        "Lower bound of the 95% bootstrap confidence interval on NDCG@10. Overlapping "
        "intervals between two rows mean the gap is not significant."
    ),
    "ndcg@10_ci_high": "Upper bound of the 95% bootstrap CI on NDCG@10.",
    "same_subject@10": (
        "Fraction of the top 10 sharing the seed's subject. A weak sanity proxy, NOT "
        "a quality target — a same-subject-only model scores high while being useless."
    ),
    "coverage": (
        "Catalog coverage — fraction of all courses that appear in at least one "
        "top-k list. Low coverage means a few courses dominate every result."
    ),
    "diversity": (
        "Intra-list diversity — average dissimilarity among recommended courses. "
        "Higher means less redundant lists."
    ),
    "novelty": (
        "Average inverse popularity of recommendations — higher favors long-tail, "
        "less-obvious picks over crowd-pleasers."
    ),
    "fit_time_s": "Seconds to fit the technique on the full catalog (one-time cost).",
    "query_latency_ms": "Median milliseconds to answer a single recommendation query.",
    "config": "The technique's hyperparameters, logged alongside its scores.",
}

# --- the three evaluation lenses (plan §3) --------------------------------------
LENSES: list[tuple[str, str]] = [
    (
        "Cross-listing pairs (primary, automatic)",
        "A course and its cross-listed twin should rank each other near the top. "
        "This is the leaderboard's default lens (sorted by NDCG@10) — but twins share "
        "near-identical text, so it validates correctness more than quality.",
    ),
    (
        "Same-subject coherence (weak proxy)",
        "Fraction of the top-k sharing the seed's subject (`same_subject@10`). A "
        "sanity floor only — never optimized for, since a same-subject-only model "
        "scores high while being useless.",
    ),
    (
        "Judged text-query set (free-text mode)",
        "Hand-labeled natural-language queries with a few relevant courses each — the "
        "only way to score free-text search. Reported in the separate text "
        "leaderboard.",
    ),
]

# Stated wherever the leaderboard is explained — the project's central guardrail.
LEAKAGE_NOTE: str = (
    "Leakage discipline: the `Cross-Listed Course(s)` column is the ground truth, so "
    "no technique may read it as a feature. The graph model is the sole exception and "
    "is scored only on a held-out split of those edges."
)


def family_of(name: str) -> str:
    """Return the family key for a raw technique name.

    Args:
        name: A technique's ``name`` (e.g. ``"sbert(all_minilm_l6_v2,idx=flat)"``).

    Returns:
        A key into :data:`FAMILIES` / :data:`FAMILY_DESCRIPTIONS`; ``"other"`` for an
        unrecognized prefix.
    """
    prefix = name.split("(", 1)[0].strip()
    return _FAMILY_BY_PREFIX.get(prefix, "other")


def family_label(name: str) -> str:
    """Return the human-readable family label for a raw technique name.

    Args:
        name: A technique's ``name``.

    Returns:
        The display label from :data:`FAMILIES` (e.g. ``"Semantic embeddings"``).
    """
    return FAMILIES[family_of(name)]


def metric_help(column: str) -> str | None:
    """Return the tooltip text for a leaderboard column, or ``None`` if unknown.

    Args:
        column: A leaderboard CSV column name.

    Returns:
        The plain-English definition from :data:`METRIC_GLOSSARY`, or ``None`` for a
        column with no registered help (the UI then shows no tooltip).
    """
    return METRIC_GLOSSARY.get(column)
