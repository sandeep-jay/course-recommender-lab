"""Lexical baselines: TF-IDF + cosine and Okapi BM25.

Both techniques are bag-of-words rankers over the combined title+description
signal. They are the simplest rung of the similarity ladder (recommender_plan.md
§2 Track A.1) and the correctness floor for everything after: near-duplicate
cross-listed twins share almost identical text, so a working lexical model nails
the primary cross-listing lens trivially.

Math sketch
-----------
*TF-IDF + cosine.* Each document is a TF-IDF vector; L2-normalized, the cosine
similarity of two documents is their dot product. Fit is one sparse
``fit_transform``; a query scores against the whole catalog as one sparse
mat-vec, ``O(nnz)``.

*BM25 (Okapi).* A probabilistic relevance score that saturates term frequency
(via ``k1``) and normalizes by document length (via ``b``):

    score(d, q) = Σ_{t∈q} idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))

with ``idf(t) = ln(1 + (N − df(t) + 0.5)/(df(t) + 0.5))``. We fold the entire
per-document factor into a precomputed sparse doc-term weight matrix ``W`` so a
query is again a single sparse mat-vec against a binary presence vector.

When it wins / loses
--------------------
Wins on exact vocabulary overlap (cross-listings, jargon-heavy fields); cheap,
interpretable, no training. Loses on synonymy and paraphrase ("ML" vs "machine
learning") — that is what the semantic-vector phase exists to beat.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from courserec.config import ARTIFACTS_DIR
from courserec.interfaces import Rec, Recommender

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """Turn a technique ``name`` into a filesystem-safe artifact directory slug."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


def bm25_weight_matrix(
    counts: sp.csr_matrix, k1: float = 1.5, b: float = 0.75
) -> sp.csr_matrix:
    """Fold the Okapi BM25 document factor into a sparse doc-term weight matrix.

    Given a document-term count matrix, returns a matrix ``W`` such that
    ``W @ q`` (with ``q`` a binary query-term presence vector) is the BM25 score
    of every document for that query. Only the count matrix's nonzeros are
    touched, so the result stays sparse.

    Args:
        counts: Document-term count matrix (``n_docs`` x ``n_terms``), CSR.
        k1: Term-frequency saturation parameter (higher = less saturation).
        b: Length-normalization strength (0 = none, 1 = full).

    Returns:
        A CSR matrix of the same shape holding BM25 weights.
    """
    counts = counts.tocsr()
    n_docs = counts.shape[0]
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avgdl = doc_len.mean() if n_docs else 0.0
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    cols = counts.indices
    tf = counts.data.astype(np.float64)
    denom = tf + k1 * (1.0 - b + b * (doc_len[rows] / avgdl if avgdl else 0.0))
    data = idf[cols] * (tf * (k1 + 1.0)) / denom
    return sp.csr_matrix((data, counts.indices, counts.indptr), shape=counts.shape)


class _LexicalRecommender(Recommender):
    """Shared machinery for bag-of-words rankers (TF-IDF, BM25).

    Subclasses supply the vocabulary/vectorizer and the fitted index matrix; the
    base class handles document construction, artifact persistence, and the two
    ``recommend_*`` methods, which both reduce to one sparse mat-vec against the
    index matrix.
    """

    def __init__(
        self, *, stopwords: bool = True, ngram_max: int = 1, title_weight: int = 1
    ) -> None:
        """Configure preprocessing shared by all lexical techniques.

        Args:
            stopwords: If true, drop English stop words.
            ngram_max: Largest n-gram to include (1 = unigrams, 2 = uni+bigrams).
            title_weight: How many times to repeat the title before the
                description, boosting the title's contribution.
        """
        if title_weight < 1:
            raise ValueError("title_weight must be >= 1")
        if ngram_max < 1:
            raise ValueError("ngram_max must be >= 1")
        self.config = {
            "stopwords": stopwords,
            "ngram_max": ngram_max,
            "title_weight": title_weight,
        }
        self._vectorizer: CountVectorizer | None = None
        self._matrix: sp.csr_matrix | None = None
        self._course_ids: list[str] = []
        self._row: dict[str, int] = {}

    # -- preprocessing ---------------------------------------------------------

    def _build_docs(self, courses: pd.DataFrame) -> list[str]:
        """Build one document string per course, weighting the title.

        Sparse-text rows (missing/empty description) fall back to the repeated
        title and never produce an exception.
        """
        tw = self.config["title_weight"]
        titles = courses["title"].fillna("")
        descs = courses["description"].fillna("")
        docs: list[str] = []
        for title, desc in zip(titles, descs, strict=True):
            parts = [title] * tw
            if desc.strip():
                parts.append(desc)
            docs.append(" ".join(p for p in parts if p).strip())
        return docs

    def _vectorizer_kwargs(self) -> dict:
        """Common scikit-learn vectorizer keyword arguments from the config."""
        return {
            "stop_words": "english" if self.config["stopwords"] else None,
            "ngram_range": (1, self.config["ngram_max"]),
            "lowercase": True,
        }

    # -- backend hooks ---------------------------------------------------------

    @abstractmethod
    def _fit_backend(self, docs: list[str]) -> tuple[CountVectorizer, sp.csr_matrix]:
        """Fit the vectorizer and build the index matrix scored at query time."""
        raise NotImplementedError

    @abstractmethod
    def _query_vector(self, text: str) -> sp.csr_matrix:
        """Map a query string to the row vector dotted against the index matrix."""
        raise NotImplementedError

    # -- fit + persistence -----------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Fit the technique, loading a cached artifact when one matches.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.
        """
        self._course_ids = list(courses.index)
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        docs = self._build_docs(courses)
        fingerprint = self._fingerprint(docs)
        if self._load(fingerprint):
            logger.info("%s: loaded cached artifact", self.name)
            return
        logger.info("%s: fitting on %d documents", self.name, len(docs))
        self._vectorizer, self._matrix = self._fit_backend(docs)
        self._save(fingerprint)

    def _fingerprint(self, docs: list[str]) -> str:
        """Hash the corpus + config so a stale or mismatched cache is rejected."""
        h = hashlib.sha1()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(b"\x00".join(d.encode() for d in docs))
        return h.hexdigest()

    @property
    def _artifact_dir(self):
        return ARTIFACTS_DIR / _slug(self.name)

    def _load(self, fingerprint: str) -> bool:
        """Load a cached vectorizer + matrix if the fingerprint matches."""
        meta_path = self._artifact_dir / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return False
        with (self._artifact_dir / "vectorizer.pkl").open("rb") as fh:
            self._vectorizer = pickle.load(fh)  # noqa: S301 (local, self-written)
        self._matrix = sp.load_npz(self._artifact_dir / "matrix.npz").tocsr()
        return True

    def _save(self, fingerprint: str) -> None:
        """Persist the fitted vectorizer, index matrix, and fingerprint."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        with (self._artifact_dir / "vectorizer.pkl").open("wb") as fh:
            pickle.dump(self._vectorizer, fh)
        sp.save_npz(self._artifact_dir / "matrix.npz", self._matrix)
        meta = {"name": self.name, "config": self.config, "fingerprint": fingerprint}
        (self._artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- recommendation --------------------------------------------------------

    def _rank(self, qvec: sp.csr_matrix, k: int, exclude_row: int | None) -> list[Rec]:
        """Score the catalog against a query vector and return the top-k recs.

        Args:
            qvec: A (1 x n_terms) sparse query vector.
            k: Maximum number of recommendations.
            exclude_row: Row index to drop from results (the seed), or ``None``.

        Returns:
            Up to ``k`` :class:`Rec` with score > 0, sorted by descending score.
        """
        scores = np.asarray((self._matrix @ qvec.T).todense()).ravel()
        if exclude_row is not None:
            scores[exclude_row] = -np.inf
        n_pos = int((scores > 0).sum())
        if n_pos == 0:
            return []
        top = min(k, n_pos)
        idx = np.argpartition(scores, -top)[-top:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [Rec(self._course_ids[i], float(scores[i])) for i in idx]

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses similar to a seed, excluding the seed itself."""
        if course_id not in self._row:
            raise KeyError(f"unknown course_id: {course_id!r}")
        row = self._row[course_id]
        seed_text = self._seed_text(row)
        return self._rank(self._query_vector(seed_text), k, exclude_row=row)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses matching a free-text query."""
        return self._rank(self._query_vector(query), k, exclude_row=None)

    def _seed_text(self, row: int) -> str:
        """Reconstruct a seed document's text from the vectorizer's analysis."""
        # The seed's own tokens drive an item-to-item query; recover them from
        # the fitted vectorizer's inverse transform so TF-IDF and BM25 share one
        # code path regardless of which matrix is stored.
        terms = self._vectorizer.inverse_transform(self._counts_row(row))[0]
        return " ".join(terms)

    @abstractmethod
    def _counts_row(self, row: int) -> sp.csr_matrix:
        """Return the seed row's term presence/counts for query reconstruction."""
        raise NotImplementedError


class TfidfRecommender(_LexicalRecommender):
    """TF-IDF vectors compared by cosine similarity."""

    def __init__(self, **kwargs) -> None:
        """See :class:`_LexicalRecommender` for the shared preprocessing config."""
        super().__init__(**kwargs)
        c = self.config
        self.name = (
            f"tfidf(sw={'on' if c['stopwords'] else 'off'},"
            f"ng=1-{c['ngram_max']},tw={c['title_weight']})"
        )

    def _fit_backend(self, docs: list[str]) -> tuple[TfidfVectorizer, sp.csr_matrix]:
        vec = TfidfVectorizer(**self._vectorizer_kwargs())
        matrix = vec.fit_transform(docs)  # L2-normalized rows
        return vec, matrix.tocsr()

    def _query_vector(self, text: str) -> sp.csr_matrix:
        # Transform reproduces the stored (normalized) row, so a dot product is
        # exactly cosine similarity.
        return self._vectorizer.transform([text])

    def _counts_row(self, row: int) -> sp.csr_matrix:
        return self._matrix[row]


class BM25Recommender(_LexicalRecommender):
    """Okapi BM25 over a precomputed sparse doc-term weight matrix."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, **kwargs) -> None:
        """Configure BM25 on top of the shared preprocessing config.

        Args:
            k1: Term-frequency saturation parameter.
            b: Length-normalization strength.
            **kwargs: Shared preprocessing options (see
                :class:`_LexicalRecommender`).
        """
        super().__init__(**kwargs)
        self.config.update({"k1": k1, "b": b})
        c = self.config
        self.name = (
            f"bm25(sw={'on' if c['stopwords'] else 'off'},"
            f"ng=1-{c['ngram_max']},tw={c['title_weight']},k1={k1},b={b})"
        )
        self._counts: sp.csr_matrix | None = None

    def _fit_backend(self, docs: list[str]) -> tuple[CountVectorizer, sp.csr_matrix]:
        vec = CountVectorizer(**self._vectorizer_kwargs())
        self._counts = vec.fit_transform(docs).tocsr()
        weights = bm25_weight_matrix(self._counts, self.config["k1"], self.config["b"])
        return vec, weights

    def _query_vector(self, text: str) -> sp.csr_matrix:
        # BM25 scores by query-term presence; the doc-side weights already carry
        # idf and length normalization.
        counts = self._vectorizer.transform([text])
        return (counts > 0).astype(np.float64)

    def _counts_row(self, row: int) -> sp.csr_matrix:
        # Recover the seed's raw counts for inverse_transform. When loaded from
        # cache the count matrix is absent, so fall back to the weight matrix,
        # which has identical sparsity (same nonzero terms).
        source = self._counts if self._counts is not None else self._matrix
        return source[row]
