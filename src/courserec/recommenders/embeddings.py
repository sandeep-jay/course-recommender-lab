"""Semantic-vector recommenders: SBERT (local) and API embeddings.

Where the lexical rung compares raw term overlap and the topic rung compares a
handful of latent axes, the semantic rung embeds each course into a dense vector
from a *pretrained* neural encoder and compares those. The encoder has already
learned, from far more text than this catalog, that "ML" ≈ "machine learning" and
that "designing deep networks" is about the same thing as "neural network
architecture" — so paraphrase and synonymy that sink lexical methods are handled
in the representation itself. This is the third rung of Track A
(recommender_plan.md §2.3, §5 Phase 3).

Two backends, one interface
---------------------------
*SBERT (local).* A Sentence-Transformers bi-encoder (``all-MiniLM-L6-v2`` and the
larger ``all-mpnet-base-v2``) runs on-device (MPS on Apple Silicon, else CPU). No
API key, no network at query time once weights are cached — the repo's local-only
guarantee (plan §1).

*API embeddings.* The same machinery against a hosted embedding model (e.g.
OpenAI ``text-embedding-3-small``), logging token count and dollar cost. It
**degrades gracefully**: with no SDK installed or no key in the environment, it
raises :class:`EmbeddingsUnavailable`, which the eval harness catches to skip +
flag the technique — never a hard failure of the suite (plan §1, rules/eval.md).

Math sketch
-----------
A document/query is encoded to a vector and L2-normalized, so cosine similarity
is a plain inner product. Search is a FAISS ``IndexFlatIP`` (exact inner-product;
at ~11k courses brute force is instant, and exactness keeps the eval
deterministic — an approximate HNSW index is the ``index_type="hnsw"`` option,
included for the ANN learning the plan calls out, not the default).

Complexity
----------
Fit is dominated by the forward pass: ``O(n_docs)`` encodes, batched, plus an
``O(n_docs · d)`` index build (``d`` = 384 for MiniLM, 768 for MPNet). A query is
one encode plus an ``O(n_docs · d)`` exact search (or sub-linear under HNSW).

Caching (two layers, both gitignored)
-------------------------------------
1. **Per-text embedding cache** keyed by ``sha1(model_name + normalized_text)``
   (rules/recommenders.md) at ``artifacts/embcache/<model>/`` — shared across
   configs and runs, so a text is embedded at most once per model, ever.
2. **Fitted-recommender artifact** at ``artifacts/<name>/``: the normalized doc
   matrix, ``course_ids``, the FAISS index, and a corpus fingerprint, so a warm
   run reloads instead of re-encoding.

When it wins / loses
--------------------
Wins where meaning outruns wording: free-text queries and conceptually-related
courses that share few exact terms — exactly the regime the judged-query lens
(plan §3 lens 3) measures and where topic models at low rank collapsed. On the
cross-listing lens it should roughly match lexical (near-duplicate twins are
already trivial in term space), so the semantic payoff shows up in free-text
ranking, not the primary automatic metric. Costs: a heavyweight dependency
(torch), slower fit than a sparse vectorizer, and embeddings that are opaque
compared to a topic–term table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from abc import abstractmethod

import numpy as np
import pandas as pd

from courserec.config import ARTIFACTS_DIR, RANDOM_SEED
from courserec.interfaces import Rec, Recommender

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: Shared per-text embedding cache root (layer 1). Gitignored with artifacts/.
_EMBCACHE_DIR = ARTIFACTS_DIR / "embcache"


class EmbeddingsUnavailable(RuntimeError):
    """Raised when a backend cannot run (missing dependency, key, or network).

    The eval harness catches this to skip and flag the technique rather than
    fail the whole suite — the graceful-degradation contract for API-backed
    techniques (rules/recommenders.md, plan §1).
    """


def _slug(name: str) -> str:
    """Turn a technique or model ``name`` into a filesystem-safe slug."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


def _normalize_text(text: str) -> str:
    """Canonicalize text for the embedding cache key and the encoder input.

    Collapses runs of whitespace and strips ends so that cosmetically different
    spellings of the same document hash to the same cache key. Case is preserved
    — the sentence encoders are cased and meaning can ride on it.

    Args:
        text: Raw document or query text.

    Returns:
        The whitespace-normalized text (never ``None``; empty stays empty).
    """
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm so a later inner product equals cosine.

    Zero/denormal-norm rows are left as-is rather than divided by a tiny number
    (which would overflow to ±inf and poison later matmuls with NaNs); their
    similarity to everything is then ~0, the intended meaning for an empty doc.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (matrix / norms).astype(np.float32)


class _EmbeddingCache:
    """Per-text embedding cache keyed by ``sha1(model_name + normalized_text)``.

    Layer 1 of the caching design: a content-addressed store shared across every
    config and run of one model, so any given text is sent through the encoder at
    most once, ever. Backed by a single ``vectors.npy`` matrix plus a row-aligned
    ``keys.json`` of hashes.
    """

    def __init__(self, model_key: str) -> None:
        """Open (or initialize) the cache for a given model identity.

        Args:
            model_key: A stable string identifying the model (e.g. its name); it
                is folded into every hash so different models never collide.
        """
        self._model_key = model_key
        self._dir = _EMBCACHE_DIR / _slug(model_key)
        self._row: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._load()

    def _key(self, normalized_text: str) -> str:
        """Hash ``model_key + normalized_text`` to the cache key (the rule's key)."""
        h = hashlib.sha1()
        h.update(self._model_key.encode())
        h.update(b"\x00")
        h.update(normalized_text.encode())
        return h.hexdigest()

    def _load(self) -> None:
        """Load the persisted matrix + key list if present (a cold cache is fine)."""
        keys_path = self._dir / "keys.json"
        vecs_path = self._dir / "vectors.npy"
        if keys_path.exists() and vecs_path.exists():
            keys = json.loads(keys_path.read_text())
            self._vectors = np.load(vecs_path)
            self._row = {k: i for i, k in enumerate(keys)}
            logger.info(
                "embcache[%s]: loaded %d cached vectors", self._model_key, len(keys)
            )

    def _save(self) -> None:
        """Persist the matrix + key list (atomic enough for a single local writer)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        keys = [k for k, _ in sorted(self._row.items(), key=lambda kv: kv[1])]
        (self._dir / "keys.json").write_text(json.dumps(keys))
        np.save(self._dir / "vectors.npy", self._vectors)

    def get_or_compute(self, texts: list[str], embed_fn) -> np.ndarray:
        """Return embeddings for ``texts``, encoding only cache misses.

        Args:
            texts: Normalized document/query texts (row order is preserved in the
                output).
            embed_fn: Callable mapping a list of texts to a ``(len, d)`` float
                array of *raw* (un-normalized) embeddings; invoked only on the
                unique misses.

        Returns:
            A ``(len(texts), d)`` array row-aligned with ``texts``.
        """
        keys = [self._key(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in self._row}
        if missing:
            logger.info(
                "embcache[%s]: %d miss / %d total — encoding misses",
                self._model_key,
                len(missing),
                len(texts),
            )
            new_vecs = np.asarray(embed_fn(list(missing.values())), dtype=np.float32)
            self._append(list(missing), new_vecs)
            self._save()
        return np.stack([self._vectors[self._row[k]] for k in keys])

    def _append(self, new_keys: list[str], new_vecs: np.ndarray) -> None:
        """Append freshly computed rows to the in-memory matrix and key index."""
        base = len(self._row)
        for offset, key in enumerate(new_keys):
            self._row[key] = base + offset
        self._vectors = (
            new_vecs if self._vectors is None else np.vstack([self._vectors, new_vecs])
        )


class _EmbeddingRecommender(Recommender):
    """Shared machinery for dense-vector rankers (SBERT, API embeddings).

    Owns document construction, the two-layer cache, L2 normalization, the FAISS
    index, artifact persistence, and the two ``recommend_*`` methods (which both
    reduce to one inner-product search). Subclasses supply only the encoder.

    Subclass responsibilities:
        * implement ``_model_key`` (stable model identity, used in cache + name),
        * implement ``_embed`` (texts -> raw embedding matrix),
        * implement ``_ensure_available`` (raise :class:`EmbeddingsUnavailable`
          if the backend cannot run), and set ``self.name`` in ``__init__``.
    """

    def __init__(self, *, index_type: str = "flat") -> None:
        """Configure the shared search/index options.

        Args:
            index_type: ``"flat"`` for an exact inner-product index (default,
                deterministic) or ``"hnsw"`` for an approximate ANN index.

        Raises:
            ValueError: If ``index_type`` is not ``"flat"`` or ``"hnsw"``.
        """
        if index_type not in ("flat", "hnsw"):
            raise ValueError("index_type must be 'flat' or 'hnsw'")
        self.config = {"index_type": index_type}
        self._embeddings: np.ndarray | None = None
        self._index = None
        self._course_ids: list[str] = []
        self._row: dict[str, int] = {}

    # -- backend hooks ---------------------------------------------------------

    @abstractmethod
    def _model_key(self) -> str:
        """Return a stable string identifying the encoder (folded into the cache)."""
        raise NotImplementedError

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts to a ``(len, d)`` matrix of raw (un-normalized) vectors."""
        raise NotImplementedError

    @abstractmethod
    def _ensure_available(self) -> None:
        """Raise :class:`EmbeddingsUnavailable` if the backend cannot run."""
        raise NotImplementedError

    # -- fit + persistence -----------------------------------------------------

    def _docs(self, courses: pd.DataFrame) -> list[str]:
        """Build one normalized document per course (title+description, sparse-safe).

        Uses the catalog's prebuilt ``text`` field, which already falls back to
        the title for description-less rows (data.py), then normalizes whitespace.
        A pathologically empty row falls back to its ``course_id`` so the encoder
        never receives an empty string.
        """
        text = courses["text"].fillna("")
        return [
            _normalize_text(t) or _normalize_text(cid)
            for cid, t in zip(courses.index, text, strict=True)
        ]

    def fit(self, courses: pd.DataFrame) -> None:
        """Encode the catalog (using the caches) and build the search index.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.

        Raises:
            EmbeddingsUnavailable: If the backend cannot run (e.g. API without a
                key). Callers in the eval harness catch this to skip + flag.
        """
        self._ensure_available()
        self._course_ids = list(courses.index)
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        docs = self._docs(courses)
        fingerprint = self._fingerprint(docs)

        if self._load(fingerprint):
            logger.info("%s: loaded cached artifact (%d docs)", self.name, len(docs))
            return

        logger.info("%s: encoding %d documents", self.name, len(docs))
        cache = _EmbeddingCache(self._model_key())
        raw = cache.get_or_compute(docs, self._embed)
        self._embeddings = _l2_normalize_rows(raw)
        self._index = self._build_index(self._embeddings)
        logger.info(
            "%s: indexed %d × %d vectors (%s)",
            self.name,
            *self._embeddings.shape,
            self.config["index_type"],
        )
        self._save(fingerprint)

    def _build_index(self, vectors: np.ndarray):
        """Build the FAISS inner-product index over the normalized doc vectors."""
        import faiss

        dim = vectors.shape[1]
        if self.config["index_type"] == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    def _fingerprint(self, docs: list[str]) -> str:
        """Hash model + config + corpus so a stale or mismatched cache is rejected."""
        h = hashlib.sha1()
        h.update(self._model_key().encode())
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(b"\x00".join(d.encode() for d in docs))
        return h.hexdigest()

    @property
    def _artifact_dir(self):
        return ARTIFACTS_DIR / _slug(self.name)

    def _load(self, fingerprint: str) -> bool:
        """Load a cached embeddings matrix + FAISS index if the fingerprint matches."""
        import faiss

        meta_path = self._artifact_dir / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return False
        self._embeddings = np.load(self._artifact_dir / "embeddings.npy")
        self._index = faiss.read_index(str(self._artifact_dir / "index.faiss"))
        self._course_ids = json.loads(
            (self._artifact_dir / "course_ids.json").read_text()
        )
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        return True

    def _save(self, fingerprint: str) -> None:
        """Persist the normalized matrix, FAISS index, ids, and fingerprint."""
        import faiss

        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._artifact_dir / "embeddings.npy", self._embeddings)
        faiss.write_index(self._index, str(self._artifact_dir / "index.faiss"))
        (self._artifact_dir / "course_ids.json").write_text(
            json.dumps(self._course_ids)
        )
        meta = {"name": self.name, "config": self.config, "fingerprint": fingerprint}
        (self._artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("%s: saved artifact to %s", self.name, self._artifact_dir)

    # -- read access (for diagnostics) -----------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        """The fitted, L2-normalized doc matrix (rows aligned with :attr:`course_ids`).

        Exposed read-only so diagnostics (the Phase 6 clustering/UMAP map,
        ``cluster.py``) can consume the cached SBERT vectors without re-encoding
        or reaching into private state.

        Raises:
            RuntimeError: If accessed before :meth:`fit`.
        """
        if self._embeddings is None:
            raise RuntimeError(f"{self.name}: fit before accessing embeddings")
        return self._embeddings

    @property
    def course_ids(self) -> list[str]:
        """The course ids in the row order of :attr:`embeddings` (a copy)."""
        return list(self._course_ids)

    # -- recommendation --------------------------------------------------------

    def _search(self, qvec: np.ndarray, k: int, exclude: str | None) -> list[Rec]:
        """Inner-product search for the top-k, dropping ``exclude`` if present.

        Args:
            qvec: A unit-norm ``(d,)`` query vector.
            k: Maximum number of recommendations.
            exclude: A ``course_id`` to drop from results (the seed), or ``None``.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending cosine. Empty if the
            query carries no signal (an all-zero vector).
        """
        if self._index is None:
            raise RuntimeError(f"{self.name}: fit before recommending")
        if not np.any(qvec):
            return []
        # Over-fetch by one so dropping the seed still leaves up to k results.
        fetch = min(k + 1, len(self._course_ids))
        query = qvec.reshape(1, -1).astype(np.float32)
        scores, idx = self._index.search(query, fetch)
        out: list[Rec] = []
        for score, i in zip(scores[0], idx[0], strict=True):
            if i < 0:  # FAISS pads with -1 when fewer than `fetch` exist
                continue
            cid = self._course_ids[i]
            if cid == exclude:
                continue
            out.append(Rec(cid, float(score)))
            if len(out) == k:
                break
        return out

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses similar to a seed, excluding the seed itself.

        Args:
            course_id: Seed course id; must exist in the fitted catalog.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending similarity, never
            including ``course_id``.

        Raises:
            KeyError: If ``course_id`` is not in the fitted catalog.
        """
        if course_id not in self._row:
            raise KeyError(f"unknown course_id: {course_id!r}")
        seed_vec = self._embeddings[self._row[course_id]]
        return self._search(seed_vec, k, exclude=course_id)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses matching a free-text query.

        The query is encoded by the same model as the corpus, lands in the same
        space, and is scored identically.

        Args:
            query: A natural-language query.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending similarity. Empty if
            the query normalizes to nothing.
        """
        normalized = _normalize_text(query)
        if not normalized:
            return []
        qvec = _l2_normalize_rows(
            np.asarray(self._embed([normalized]), dtype=np.float32)
        )[0]
        return self._search(qvec, k, exclude=None)


class SbertRecommender(_EmbeddingRecommender):
    """Local Sentence-Transformers bi-encoder (MPS on Apple Silicon, else CPU)."""

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Configure the SBERT encoder.

        Args:
            model_name: A Sentence-Transformers model id (default
                ``all-MiniLM-L6-v2``; ``all-mpnet-base-v2`` is the larger option).
            batch_size: Encode batch size.
            device: Force a torch device (``"mps"``/``"cpu"``/``"cuda"``); default
                auto-selects MPS when available, else CPU.
            **kwargs: Shared options (see :class:`_EmbeddingRecommender`).
        """
        super().__init__(**kwargs)
        self.config.update({"model_name": model_name, "batch_size": batch_size})
        self._device = device
        self._model = None  # lazily loaded on first encode (heavy import)
        self.name = f"sbert({_slug(model_name)},idx={self.config['index_type']})"

    def _model_key(self) -> str:
        return f"sbert:{self.config['model_name']}"

    def _resolve_device(self) -> str:
        """Pick the torch device: explicit override, else MPS if present, else CPU."""
        if self._device:
            return self._device
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _ensure_available(self) -> None:
        """Verify torch + sentence-transformers import (the heavy optional extra)."""
        try:
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via skip path
            raise EmbeddingsUnavailable(
                f"{self.name}: install the 'semantic' extra "
                '(`pip install -e ".[semantic]"`)'
            ) from exc

    def _load_model(self):
        """Load the Sentence-Transformer once, pinning the seed for determinism."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.manual_seed(RANDOM_SEED)
            device = self._resolve_device()
            logger.info(
                "%s: loading %s on %s", self.name, self.config["model_name"], device
            )
            self._model = SentenceTransformer(self.config["model_name"], device=device)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts to raw vectors (normalization happens in the base class)."""
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=self.config["batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )


#: Default per-1M-token price (USD) for the API embedding model, for cost logging.
#: A magic number only in the sense that the vendor sets it; surfaced as config so
#: a price change is a one-line edit, never buried in logic.
_API_PRICE_PER_1M_TOKENS = 0.02


class ApiEmbeddingRecommender(_EmbeddingRecommender):
    """Hosted embedding model (OpenAI by default), with cost logging + graceful skip.

    Behaves identically to the SBERT backend but calls an API to embed. With no
    SDK or no ``OPENAI_API_KEY`` in the environment it raises
    :class:`EmbeddingsUnavailable` so the suite skips and flags it rather than
    failing — the repo must run local-only with no key (plan §1).
    """

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        price_per_1m_tokens: float = _API_PRICE_PER_1M_TOKENS,
        **kwargs,
    ) -> None:
        """Configure the API embedding backend.

        Args:
            model_name: Hosted embedding model id.
            api_key_env: Environment variable holding the API key (never a literal
                key — the no-hardcoded-secrets rule).
            price_per_1m_tokens: USD per 1M tokens, for cost logging.
            **kwargs: Shared options (see :class:`_EmbeddingRecommender`).
        """
        super().__init__(**kwargs)
        self.config.update({"model_name": model_name, "api_key_env": api_key_env})
        self._price_per_1m = price_per_1m_tokens
        self._tokens_used = 0
        self.name = f"api({_slug(model_name)},idx={self.config['index_type']})"

    def _model_key(self) -> str:
        return f"api:{self.config['model_name']}"

    def _ensure_available(self) -> None:
        """Require the openai SDK and a key in the environment, else skip."""
        if not os.environ.get(self.config["api_key_env"]):
            raise EmbeddingsUnavailable(
                f"{self.name}: no {self.config['api_key_env']} set — skipping "
                "(the repo runs local-only with no key)"
            )
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise EmbeddingsUnavailable(
                f"{self.name}: openai SDK not installed — skipping"
            ) from exc

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed via the API, accumulating token count + cost for the leaderboard."""
        import openai

        client = openai.OpenAI(api_key=os.environ[self.config["api_key_env"]])
        resp = client.embeddings.create(model=self.config["model_name"], input=texts)
        self._tokens_used += resp.usage.total_tokens
        cost = self._tokens_used / 1_000_000 * self._price_per_1m
        logger.info(
            "%s: %d tokens cumulative (~$%.4f)", self.name, self._tokens_used, cost
        )
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
