"""Retrieve → rerank → diversify: a two-stage pipeline with an MMR knob.

The semantic rung (``embeddings.py``) ranks the whole catalog with a single
*bi-encoder* dot product: every course and the query are embedded independently,
so scoring is cheap (one index search) but the query and a course never "see"
each other during encoding. A *cross-encoder* does the opposite — it feeds the
query and a candidate through the model **together** and reads out a relevance
score — which is far more accurate but far too slow to run over all ~11k courses
per query. The standard resolution, and this module, is two stages
(recommender_plan.md §2.4, §5 Phase 4):

1. **Retrieve.** A fast bi-encoder (any fitted :class:`_EmbeddingRecommender`,
   default :class:`SbertRecommender`) fetches the top ``retrieve_n`` (~50)
   candidates — high recall, cheap.
2. **Rerank.** A cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``) scores
   each ``(query, candidate)`` pair jointly and re-orders them — high precision,
   over a small set.

Diversity (MMR)
---------------
Pure relevance ranking clusters near-duplicates at the top (the cross-listed
twins, three flavors of the same lecture). Maximal Marginal Relevance trades
relevance against novelty with one knob ``mmr_lambda`` ∈ [0, 1]:

    MMR(c) = λ · rel(c) − (1 − λ) · max_{s ∈ selected} sim(c, s)

``rel`` is the (min-max-normalized) cross-encoder score; ``sim`` is cosine
between candidates in the **bi-encoder** space (already L2-normalized, so a dot
product). ``λ = 1`` is pure cross-encoder order (most relevant, least diverse);
lowering ``λ`` pushes the intra-list-diversity metric up — the acceptance test
for this phase (recommender_plan.md §5). The greedy MMR value is provably
non-increasing across selections, so the returned scores are sorted descending,
honoring the interface contract.

Reuse, caching, graceful degradation
------------------------------------
Retrieval reuses the base recommender wholesale — its two-layer embedding cache
and FAISS artifact (``artifacts/<base>/``) persist and reload exactly as before;
this stage adds no precomputed artifact because cross-encoder scoring is
inherently query-time. The cross-encoder is another ``sentence-transformers``
model (the ``semantic`` extra), so the whole technique **degrades gracefully**:
with the extra absent it raises :class:`EmbeddingsUnavailable`, which the eval
harness catches to skip + flag — never a hard failure (plan §1).

Complexity
----------
Per query: one bi-encoder search (``O(n_docs · d)`` exact), then ``retrieve_n``
cross-encoder forward passes, then ``O(retrieve_n² )`` MMR similarity work over
the small candidate set. The cross-encoder passes dominate wall-clock.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from courserec.config import RANDOM_SEED
from courserec.interfaces import Rec, Recommender
from courserec.recommenders.embeddings import (
    EmbeddingsUnavailable,
    SbertRecommender,
    _EmbeddingRecommender,
    _normalize_text,
    _slug,
)

logger = logging.getLogger(__name__)

#: Default cross-encoder: a small MS-MARCO reranker, part of the 'semantic' extra.
_DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRecommender(Recommender):
    """Bi-encoder retrieval + cross-encoder rerank + MMR diversity (Phase 4).

    Wraps a fitted base retriever and adds a query-time rerank/diversify stage.
    Honors the full :class:`Recommender` contract: excludes the seed, returns
    ``list[Rec]`` of length ``<= k`` sorted by descending score, falls back to
    the title on sparse text (inherited from the base), and serves both
    ``recommend_similar`` and ``recommend_by_text``.
    """

    def __init__(
        self,
        *,
        base: _EmbeddingRecommender | None = None,
        cross_encoder: str = _DEFAULT_CROSS_ENCODER,
        retrieve_n: int = 50,
        mmr_lambda: float = 0.5,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Configure the two-stage pipeline.

        Args:
            base: The first-stage retriever (any fitted-capable
                :class:`_EmbeddingRecommender`). Defaults to a MiniLM
                :class:`SbertRecommender`.
            cross_encoder: A ``sentence-transformers`` CrossEncoder model id.
            retrieve_n: How many candidates the base retrieves before reranking.
            mmr_lambda: MMR relevance/diversity trade-off in ``[0, 1]``; ``1`` is
                pure cross-encoder relevance, lower values add diversity.
            batch_size: Cross-encoder scoring batch size.
            device: Force a torch device; default auto-selects MPS else CPU.

        Raises:
            ValueError: If ``mmr_lambda`` is outside ``[0, 1]`` or
                ``retrieve_n`` is not positive.
        """
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if retrieve_n < 1:
            raise ValueError("retrieve_n must be >= 1")
        self._base = base if base is not None else SbertRecommender()
        self._cross_encoder_name = cross_encoder
        self._batch_size = batch_size
        self._device = device
        self._model = None  # lazily loaded on first rerank (heavy import)
        self._text_by_id: dict[str, str] = {}
        self.config = {
            "base": self._base.name,
            "cross_encoder": cross_encoder,
            "retrieve_n": retrieve_n,
            "mmr_lambda": mmr_lambda,
        }
        self.name = (
            f"rerank({_slug(cross_encoder)},base={_slug(self._base.name)},"
            f"n={retrieve_n},mmr={mmr_lambda})"
        )

    # -- availability + model loading -----------------------------------------

    def _ensure_available(self) -> None:
        """Raise :class:`EmbeddingsUnavailable` if the cross-encoder can't run."""
        try:
            import torch  # noqa: F401
            from sentence_transformers import CrossEncoder  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via skip path
            raise EmbeddingsUnavailable(
                f"{self.name}: install the 'semantic' extra "
                '(`pip install -e ".[semantic]"`)'
            ) from exc

    def _resolve_device(self) -> str:
        """Pick the torch device: explicit override, else MPS if present, else CPU."""
        if self._device:
            return self._device
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_model(self):
        """Load the CrossEncoder once, pinning the seed for determinism."""
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            torch.manual_seed(RANDOM_SEED)
            device = self._resolve_device()
            logger.info(
                "%s: loading %s on %s", self.name, self._cross_encoder_name, device
            )
            self._model = CrossEncoder(self._cross_encoder_name, device=device)
        return self._model

    # -- fit ------------------------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Fit the base retriever and capture candidate texts for reranking.

        The base persists/reloads its own embedding artifact; this stage adds no
        precomputed artifact (cross-encoder scoring is query-time). The
        cross-encoder weights are loaded lazily on first use.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.

        Raises:
            EmbeddingsUnavailable: If the cross-encoder or base backend cannot
                run (e.g. the ``semantic`` extra is not installed).
        """
        self._ensure_available()
        self._base.fit(courses)
        text = courses["text"].fillna("")
        self._text_by_id = {
            cid: _normalize_text(t) or _normalize_text(cid)
            for cid, t in zip(courses.index, text, strict=True)
        }

    # -- reranking core -------------------------------------------------------

    def _cross_scores(self, query: str, candidates: list[str]) -> np.ndarray:
        """Score each ``(query, candidate-text)`` pair with the cross-encoder."""
        model = self._load_model()
        pairs = [[query, self._text_by_id[c]] for c in candidates]
        scores = model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float64)

    def _candidate_vectors(self, candidates: list[str]) -> np.ndarray:
        """Fetch the base's L2-normalized embeddings for the candidates (for MMR)."""
        rows = [self._base._row[c] for c in candidates]
        return self._base._embeddings[rows]

    def _mmr(
        self, candidates: list[str], rel: np.ndarray, vectors: np.ndarray, k: int
    ) -> list[Rec]:
        """Greedily select up to ``k`` candidates by Maximal Marginal Relevance.

        Args:
            candidates: Candidate ``course_id``s, in retrieval order.
            rel: Per-candidate relevance (cross-encoder score), min-max normalized.
            vectors: Per-candidate L2-normalized embeddings (cosine via dot).
            k: Maximum number to select.

        Returns:
            Up to ``k`` :class:`Rec`, scored by the (non-increasing) MMR value so
            the list is sorted by descending score.
        """
        lam = self.config["mmr_lambda"]
        # Pairwise cosine over the small candidate set (vectors are L2-normalized).
        # errstate guards a spurious numpy SIMD "divide by zero in matmul"
        # RuntimeWarning (no division actually occurs in a matmul).
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = vectors @ vectors.T
        n = len(candidates)
        remaining = list(range(n))
        selected: list[int] = []
        out: list[Rec] = []
        while remaining and len(selected) < k:
            if selected:
                # max similarity of each remaining item to any chosen item
                max_sim = sims[np.ix_(remaining, selected)].max(axis=1)
            else:
                max_sim = np.zeros(len(remaining))
            mmr = lam * rel[remaining] - (1.0 - lam) * max_sim
            best = int(np.argmax(mmr))
            chosen = remaining.pop(best)
            selected.append(chosen)
            out.append(Rec(candidates[chosen], float(mmr[best])))
        return out

    def _rerank(self, query: str, candidates: list[str], k: int) -> list[Rec]:
        """Run the cross-encoder + MMR over retrieved candidates for one query."""
        if not candidates:
            return []
        rel = self._cross_scores(query, candidates)
        # Min-max normalize relevance into [0, 1] so it is comparable to the
        # cosine similarity term in MMR regardless of the cross-encoder's scale.
        span = rel.max() - rel.min()
        rel = (rel - rel.min()) / span if span > 0 else np.zeros_like(rel)
        vectors = self._candidate_vectors(candidates)
        return self._mmr(candidates, rel, vectors, k)

    # -- recommendation -------------------------------------------------------

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses similar to a seed, reranked and diversified.

        The base retrieves ``retrieve_n`` candidates (seed already excluded); the
        cross-encoder rescores them against the seed's own text, then MMR
        re-orders for diversity.

        Args:
            course_id: Seed course id; must exist in the fitted catalog.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending score, never including
            ``course_id``.

        Raises:
            KeyError: If ``course_id`` is not in the fitted catalog.
        """
        if course_id not in self._text_by_id:
            raise KeyError(f"unknown course_id: {course_id!r}")
        retrieved = self._base.recommend_similar(course_id, k=self.config["retrieve_n"])
        candidates = [r.course_id for r in retrieved]
        return self._rerank(self._text_by_id[course_id], candidates, k)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses matching a free-text query, reranked and diversified.

        Args:
            query: A natural-language query.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending score. Empty if the
            query normalizes to nothing.
        """
        normalized = _normalize_text(query)
        if not normalized:
            return []
        retrieved = self._base.recommend_by_text(query, k=self.config["retrieve_n"])
        candidates = [r.course_id for r in retrieved]
        return self._rerank(normalized, candidates, k)
