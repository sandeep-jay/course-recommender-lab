"""Metadata fusion: a weighted blend of structured facets and a text vector.

Every rung so far compares courses on *one* signal — words (lexical), latent
word axes (topics), or learned meaning (semantic). But the catalog also carries
cheap, perfectly-clean structure the text ignores: a course's subject,
department, level band, and unit count. Two courses can read very differently yet
both be graduate Mechanical Engineering seminars; two can share vocabulary yet sit
in different divisions. This technique fuses the two views — a TF-IDF text block
and a one-hot metadata block — into one vector and ranks by their blend. It is the
first rung of Track B (recommender_plan.md §2.5, §5 Phase 5 / B.5).

Why TF-IDF (not SBERT) as the text half
----------------------------------------
The text backend is TF-IDF, so the whole vector stays sparse, the technique needs
no heavy optional extra or API key (it always runs, plan §1), and — most usefully
— it makes the metadata contribution legible: ``metadata(...)`` differs from the
``tfidf(...)`` baseline already on the leaderboard *only* by the fused metadata
block, so any lift is attributable to the structure, not the encoder. Fusing
metadata onto SBERT instead is a natural extension, left out to keep this a clean
ablation.

Math sketch
-----------
Each course becomes two L2-normalized blocks: a TF-IDF text vector ``t`` and a
one-hot metadata vector ``m`` (subject, department, level, units — each an active
indicator). We scale them by a single fusion knob ``λ = text_weight`` and its
complement and concatenate::

    v(d) = [ λ · t(d)  ‖  (1 − λ) · m(d) ]

Because each block is unit-norm, the inner product of two such vectors is a
weighted sum of two cosines::

    score(d, q) = λ² · cos_text(d, q) + (1 − λ)² · cos_meta(d, q)

so ``λ = 1`` recovers pure TF-IDF, ``λ = 0`` is pure metadata, and the sweep
between tunes how much the structure pulls. A free-text query has no facets, so
its metadata block is zero and ``recommend_by_text`` degrades to pure text
ranking — exactly the desired behavior (a query string has no department).

A standout property: a course with a one-word or missing description has a nearly
empty text block, but its metadata block is fully populated, so fusion still
ranks it sensibly where a text-only method would flounder.

Complexity & caching
--------------------
Fit is one TF-IDF ``fit_transform`` plus a one-hot encode, both ``O(nnz)``; a
query is a single sparse mat-vec against the fused matrix. The fitted artifact
(fused matrix, text vectorizer, metadata width, ids, fingerprint) persists to
``artifacts/<name>/`` and reloads instead of recomputing (plan §1, rules).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from courserec.config import ARTIFACTS_DIR
from courserec.interfaces import Rec, Recommender

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: Structured columns fused as one-hot facets, in a fixed order for a stable name.
#: ``units`` is keyed off ``units_min``; the rest are categorical columns. Any
#: facet whose column is absent from the catalog is silently skipped, so the
#: technique runs on a metadata-poor frame (e.g. a test catalog) without crashing.
_DEFAULT_FACETS: tuple[str, ...] = ("subject", "department", "level", "units")


def _slug(name: str) -> str:
    """Turn a technique ``name`` into a filesystem-safe artifact directory slug."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


class MetadataRecommender(Recommender):
    """Fuse a weighted one-hot metadata block with a TF-IDF text block.

    The single hyperparameter that matters is ``text_weight`` (λ): the relative
    pull of words versus structure. Both ``recommend_*`` methods reduce to one
    sparse mat-vec against the fused, per-block-normalized matrix.
    """

    def __init__(
        self,
        *,
        text_weight: float = 0.7,
        facets: tuple[str, ...] = _DEFAULT_FACETS,
        stopwords: bool = True,
        ngram_max: int = 1,
        title_weight: int = 1,
    ) -> None:
        """Configure the fusion blend and the shared text preprocessing.

        Args:
            text_weight: Fusion knob ``λ`` in ``[0, 1]``. The text block is scaled
                by ``λ`` and the metadata block by ``1 − λ``; ``1.0`` is pure
                TF-IDF, ``0.0`` is pure metadata.
            facets: Structured columns to one-hot encode (absent columns are
                skipped). ``"units"`` reads ``units_min``; others are categorical.
            stopwords: If true, drop English stop words from the text block.
            ngram_max: Largest n-gram in the text block (1 = unigrams).
            title_weight: How many times to repeat the title before the
                description, boosting the title's contribution to the text block.

        Raises:
            ValueError: If ``text_weight`` is outside ``[0, 1]`` or the
                preprocessing knobs are below their minimums.
        """
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be in [0, 1]")
        if title_weight < 1:
            raise ValueError("title_weight must be >= 1")
        if ngram_max < 1:
            raise ValueError("ngram_max must be >= 1")
        self.config = {
            "text_weight": text_weight,
            "facets": list(facets),
            "stopwords": stopwords,
            "ngram_max": ngram_max,
            "title_weight": title_weight,
        }
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: sp.csr_matrix | None = None
        self._n_meta: int = 0
        self._course_ids: list[str] = []
        self._row: dict[str, int] = {}
        self.name = f"metadata(text={text_weight:g},facets={'+'.join(facets)})"

    # -- feature construction --------------------------------------------------

    def _build_docs(self, courses: pd.DataFrame) -> list[str]:
        """Build one title-weighted text document per course (sparse-text safe).

        Missing/empty descriptions fall back to the repeated title, so a course
        with no description still produces a (title-only) text block rather than
        crashing.
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

    def _facet_dicts(self, courses: pd.DataFrame) -> list[dict[str, int]]:
        """Map each course to a ``{facet=value: 1}`` dict for one-hot encoding.

        Only configured facets whose column is present and value is non-null
        contribute, so a metadata-poor catalog yields smaller dicts (and an
        all-missing course yields an empty dict → a zero metadata block) rather
        than an error. ``units`` is bucketed by its raw ``units_min`` value so the
        indicator is shared by all courses of that unit count.
        """
        facets = self.config["facets"]
        dicts: list[dict[str, int]] = []
        for cid in courses.index:
            row = courses.loc[cid]
            features: dict[str, int] = {}
            for facet in facets:
                value = self._facet_value(facet, row)
                if value is not None:
                    features[f"{facet}={value}"] = 1
            dicts.append(features)
        return dicts

    @staticmethod
    def _facet_value(facet: str, row: pd.Series) -> str | None:
        """Read one facet's categorical value from a course row, or ``None``.

        ``units`` reads the numeric ``units_min`` and renders it as a compact
        string label; every other facet is a categorical column read verbatim.
        Missing columns and null values both yield ``None`` (skip the facet).
        """
        column = "units_min" if facet == "units" else facet
        if column not in row.index:
            return None
        value = row[column]
        if pd.isna(value):
            return None
        return f"{value:g}" if facet == "units" else str(value)

    # -- fit + persistence -----------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Fit the text + metadata blocks and fuse them, loading a cache if matched.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.
        """
        self._course_ids = list(courses.index)
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        docs = self._build_docs(courses)
        facet_dicts = self._facet_dicts(courses)
        fingerprint = self._fingerprint(docs, facet_dicts)
        if self._load(fingerprint):
            logger.info("%s: loaded cached artifact", self.name)
            return

        logger.info("%s: fusing %d docs + metadata", self.name, len(docs))
        self._vectorizer = TfidfVectorizer(
            stop_words="english" if self.config["stopwords"] else None,
            ngram_range=(1, self.config["ngram_max"]),
            lowercase=True,
        )
        text = self._vectorizer.fit_transform(docs)  # rows already L2-normalized
        meta = normalize(DictVectorizer(sparse=True).fit_transform(facet_dicts))
        self._n_meta = meta.shape[1]
        self._matrix = self._fuse(text, meta).tocsr()
        logger.info(
            "%s: fused %d × (%d text + %d meta)",
            self.name,
            self._matrix.shape[0],
            text.shape[1],
            self._n_meta,
        )
        self._save(fingerprint)

    def _fuse(self, text: sp.spmatrix, meta: sp.spmatrix) -> sp.spmatrix:
        """Scale each unit-norm block by the fusion knob and concatenate them."""
        lam = self.config["text_weight"]
        return sp.hstack([text * lam, meta * (1.0 - lam)])

    def _fingerprint(self, docs: list[str], facet_dicts: list[dict]) -> str:
        """Hash config + text corpus + facet values so a stale cache is rejected."""
        h = hashlib.sha1()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(b"\x00".join(d.encode() for d in docs))
        h.update(json.dumps(facet_dicts, sort_keys=True).encode())
        return h.hexdigest()

    @property
    def _artifact_dir(self):
        return ARTIFACTS_DIR / _slug(self.name)

    def _load(self, fingerprint: str) -> bool:
        """Load a cached vectorizer + fused matrix if the fingerprint matches."""
        meta_path = self._artifact_dir / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return False
        with (self._artifact_dir / "vectorizer.pkl").open("rb") as fh:
            self._vectorizer = pickle.load(fh)  # noqa: S301 (local, self-written)
        self._matrix = sp.load_npz(self._artifact_dir / "matrix.npz").tocsr()
        self._n_meta = int(meta["n_meta"])
        return True

    def _save(self, fingerprint: str) -> None:
        """Persist the fitted vectorizer, fused matrix, metadata width, fingerprint."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        with (self._artifact_dir / "vectorizer.pkl").open("wb") as fh:
            pickle.dump(self._vectorizer, fh)
        sp.save_npz(self._artifact_dir / "matrix.npz", self._matrix)
        meta = {
            "name": self.name,
            "config": self.config,
            "n_meta": self._n_meta,
            "fingerprint": fingerprint,
        }
        (self._artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- recommendation --------------------------------------------------------

    def _rank(self, qvec: sp.spmatrix, k: int, exclude_row: int | None) -> list[Rec]:
        """Score the catalog against a fused query vector and return the top-k recs.

        Args:
            qvec: A ``(1, n_text + n_meta)`` sparse fused query vector.
            k: Maximum number of recommendations.
            exclude_row: Row index to drop from results (the seed), or ``None``.

        Returns:
            Up to ``k`` :class:`Rec` with score > 0, sorted by descending score.
        """
        if self._matrix is None:
            raise RuntimeError(f"{self.name}: fit before recommending")
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
        """Recommend courses similar to a seed (text + metadata), excluding the seed.

        The seed's own fused row is reused as the query, so its metadata block
        participates in the match — unlike a free-text query.

        Args:
            course_id: Seed course id; must exist in the fitted catalog.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending fused similarity, never
            including ``course_id``.

        Raises:
            KeyError: If ``course_id`` is not in the fitted catalog.
        """
        if course_id not in self._row:
            raise KeyError(f"unknown course_id: {course_id!r}")
        row = self._row[course_id]
        return self._rank(self._matrix[row], k, exclude_row=row)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses matching a free-text query (text block only).

        A query carries no facets, so its metadata block is zero and ranking
        reduces to weighted TF-IDF cosine — the fused metadata helps only the
        item-to-item lens, never free text.

        Args:
            query: A natural-language query.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending score.
        """
        if self._vectorizer is None:
            raise RuntimeError(f"{self.name}: fit before recommending")
        text = self._vectorizer.transform([query]) * self.config["text_weight"]
        zeros = sp.csr_matrix((1, self._n_meta), dtype=text.dtype)
        return self._rank(sp.hstack([text, zeros]).tocsr(), k, exclude_row=None)
