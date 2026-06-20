"""LLM enrichment via local Ollama: distill each course into structured tags.

Every prior rung consumes the catalog's *given* signal — its words, its facets.
This one manufactures new signal: a local LLM reads each course's title and
description and emits structured tags (topics, skills, level, prereqs-mentioned),
turning prose into a compact, normalized feature space a recommender can compare.
It is the first piece of Track B rung 8 (recommender_plan.md §2.8 / §5 Phase 7);
the zero-shot reranker and the "why this fits" explanation are the later pieces.

Why local Ollama (not a cloud API)
----------------------------------
The model runs on-device through Ollama's HTTP API (``localhost:11434``) — no API
key, no per-call cost, no network beyond localhost. That keeps the repo's
local-only guarantee (plan §1) fully intact: the graceful-skip path becomes
"Ollama unreachable or no enrichment cached" rather than "no API key set." The
client is stdlib ``urllib`` — the technique adds **zero new dependencies**; its
only runtime needs are a running Ollama daemon and a pulled model, both checked at
runtime and skipped (with a flag) when absent (rules/recommenders.md, plan §1).

How tags become a ranking
-------------------------
Extraction is structured: Ollama's ``format`` field carries a JSON schema, so the
model returns a validated :class:`CourseTags` object, not free text to parse. Each
course's tags are flattened into a **profile string** (topics + skills +
prereqs-mentioned); the profiles are TF-IDF-vectorized and compared by cosine,
exactly like the lexical rung — but over the LLM's distilled vocabulary instead of
raw description text. The point is normalization: the encoder collapses "ML" and
"machine learning" to the same emitted topic, so conceptually-kin courses share
features even when their descriptions do not.

Enrichment is a separate, explicit step
----------------------------------------
``fit`` never calls the LLM — it only *reads* the tag cache and falls back to a
course's raw text where no tags exist. The slow generation pass lives in
:func:`enrich_courses` (driven by ``scripts/enrich_catalog.py``), so the
evaluation run stays fast and deterministic and the expensive ~N-course pass is a
deliberate, resumable, cached one-time cost. A run with no tags cached and Ollama
down skips with a flag rather than masquerading as a (useless) raw-text TF-IDF
duplicate.

Caching (gitignored)
--------------------
Extracted tags persist to ``artifacts/llmcache/<model>/tags.json`` keyed by
``sha1(model + normalized_text)`` (rules/recommenders.md) — a course is enriched
at most once per model, ever, and re-runs are free. The fitted recommender's own
TF-IDF artifact persists to ``artifacts/<name>/`` like every other rung.

When it wins / loses
--------------------
Should help conceptual / free-text matching, where the LLM's abstraction beats
surface wording. On the cross-listing lens it is unlikely to beat lexical — twins
already share near-identical text, and distillation can only blur an easy exact
match — so the honest payoff, if any, is on the judged free-text lens. Costs: a
running model, a slow enrichment pass, and tags that inherit the LLM's blind spots.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from courserec.config import ARTIFACTS_DIR, DEFAULT_LLM_MODEL, OLLAMA_HOST, RANDOM_SEED
from courserec.interfaces import Rec, Recommender
from courserec.recommenders.embeddings import SbertRecommender

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: Shared LLM tag-cache root (gitignored with artifacts/).
_LLMCACHE_DIR = ARTIFACTS_DIR / "llmcache"

#: JSON schema handed to Ollama's ``format`` field so generation is structured and
#: validated server-side — the model must return exactly these keys.
_TAG_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
        "skills": {"type": "array", "items": {"type": "string"}},
        "level": {
            "type": "string",
            "enum": ["intro", "intermediate", "advanced", "graduate"],
        },
        "prereqs_mentioned": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topics", "skills", "level", "prereqs_mentioned"],
}

_EXTRACT_INSTRUCTION = (
    "Extract structured metadata from this university course for a recommender "
    "system. topics = the subjects it covers; skills = what a student can do "
    "after; level = its difficulty band; prereqs_mentioned = any prior knowledge "
    "the description names. Be concise and return ONLY the JSON."
)

#: JSON schema for the zero-shot reranker (§2.8b): the model returns an ordered
#: list of candidate numbers (1-based, referencing the numbered prompt listing),
#: validated server-side so the reply is a permutation, not prose to parse.
_RERANK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ranking": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ranking"],
}

_RERANK_INSTRUCTION = (
    "You are ranking university courses by how well each matches a query, for a "
    "recommender system. Read the query, then the numbered candidate courses. "
    "Return the candidate numbers ordered from most to least relevant to the "
    "query, including every number exactly once. Return ONLY the JSON."
)

#: JSON schema for the "why this fits" explainer (§2.8c): the model returns one
#: short justification string, validated server-side so the reply is the reason
#: and nothing else (no preamble, no thinking).
_EXPLAIN_SCHEMA: dict = {
    "type": "object",
    "properties": {"reason": {"type": "string"}},
    "required": ["reason"],
}

_EXPLAIN_INSTRUCTION = (
    "You explain course recommendations for a catalog UI. Given a query (a seed "
    "course's text, or a free-text search) and one recommended course, write a "
    "single short sentence (at most ~25 words) saying why the recommended course "
    "fits the query. Name the concrete shared topic or skill; do not restate the "
    "title verbatim, hedge, or add a preamble. Return ONLY the JSON."
)


def _slug(name: str) -> str:
    """Turn a technique or model ``name`` into a filesystem-safe slug."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


def _normalize_text(text: str) -> str:
    """Collapse whitespace so cosmetically different spellings share a cache key."""
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


class LLMUnavailable(RuntimeError):
    """Raised when the rung cannot run (Ollama unreachable, or no tags cached).

    The eval harness catches this to skip and flag the technique rather than fail
    the whole suite — the graceful-degradation contract for LLM/API-backed
    techniques (rules/recommenders.md, plan §1).
    """


@dataclass(frozen=True)
class CourseTags:
    """Structured tags an LLM extracted from one course's title + description.

    Attributes:
        topics: Subjects the course covers (the most discriminative signal).
        skills: Capabilities a student gains.
        level: Difficulty band (``intro``/``intermediate``/``advanced``/``graduate``).
        prereqs_mentioned: Prior knowledge the description names (not the real
            prereq graph, which the catalog lacks).
    """

    topics: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    level: str = ""
    prereqs_mentioned: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> CourseTags:
        """Build from a parsed JSON object, tolerating missing/extra keys."""
        return cls(
            topics=[str(t) for t in payload.get("topics", [])],
            skills=[str(s) for s in payload.get("skills", [])],
            level=str(payload.get("level", "")),
            prereqs_mentioned=[str(p) for p in payload.get("prereqs_mentioned", [])],
        )

    def to_dict(self) -> dict:
        """Serialize back to a plain JSON-able dict for the cache."""
        return {
            "topics": self.topics,
            "skills": self.skills,
            "level": self.level,
            "prereqs_mentioned": self.prereqs_mentioned,
        }

    def profile_text(self) -> str:
        """Flatten the conceptual tags into the string the TF-IDF space consumes.

        Topics + skills + prereqs-mentioned are the "what is this course about"
        signal; ``level`` is intentionally excluded (a coarse 4-way band already
        covered by the metadata rung's facets). Empty for a tag-less course.
        """
        return " ".join([*self.topics, *self.skills, *self.prereqs_mentioned]).strip()


# --------------------------------------------------------------------------- #
# Ollama client (stdlib urllib — no new dependency)                           #
# --------------------------------------------------------------------------- #


class OllamaClient:
    """Thin client over the local Ollama HTTP API for structured tag extraction.

    Determinism is pinned (``temperature=0``, ``seed=RANDOM_SEED``) and "thinking"
    is disabled so the response is the JSON object and nothing else.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        host: str = OLLAMA_HOST,
        timeout: float = 120.0,
    ) -> None:
        """Configure the endpoint and model.

        Args:
            model: The Ollama model tag (e.g. ``"qwen3:8b"``).
            host: Base URL of the Ollama daemon.
            timeout: Per-request timeout in seconds.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        """POST ``payload`` as JSON to ``host + path`` and return the parsed reply."""
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read())

    def available(self) -> bool:
        """Return True if the daemon is reachable and the model is pulled."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                tags = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("Ollama not reachable at %s: %s", self.host, exc)
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names:
            logger.warning(
                "Ollama model %r not pulled (have: %s)", self.model, sorted(names)
            )
            return False
        return True

    def extract_tags(self, title: str, text: str) -> CourseTags:
        """Extract structured tags for one course via a deterministic generation.

        Args:
            title: Course title.
            text: Combined course text (title + description).

        Returns:
            The parsed :class:`CourseTags`.

        Raises:
            LLMUnavailable: If the request fails or the reply is not valid JSON.
        """
        prompt = f"{_EXTRACT_INSTRUCTION}\n\nTitle: {title}\nDescription: {text}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": _TAG_SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": RANDOM_SEED},
        }
        try:
            reply = self._post("/api/generate", payload)
            return CourseTags.from_dict(json.loads(reply["response"]))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            raise LLMUnavailable(f"Ollama generation failed: {exc}") from exc

    def rank_candidates(self, query: str, candidate_texts: list[str]) -> list[int]:
        """Rank candidate courses for a query, returning their 0-based positions.

        The candidates are presented to the model as a numbered listing; it
        returns those numbers reordered most- to least-relevant. The returned
        indices are *as the model gave them* — possibly partial, duplicated, or
        out of range; the caller reconciles them against the candidate set.

        Args:
            query: The seed course's text (item-to-item mode) or a free-text query.
            candidate_texts: The retrieved candidates' full texts, in base order.

        Returns:
            The model's ranking as 0-based indices into ``candidate_texts``.

        Raises:
            LLMUnavailable: If the request fails or the reply is not valid JSON.
        """
        listing = "\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(candidate_texts)
        )
        prompt = f"{_RERANK_INSTRUCTION}\n\nQuery: {query}\n\nCandidates:\n{listing}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": _RERANK_SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": RANDOM_SEED},
        }
        try:
            reply = self._post("/api/generate", payload)
            ranking = json.loads(reply["response"])["ranking"]
            return [int(n) - 1 for n in ranking]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailable(f"Ollama rerank failed: {exc}") from exc

    def explain(self, query: str, candidate_title: str, candidate_text: str) -> str:
        """Generate a one-sentence "why this fits" justification for one candidate.

        Args:
            query: The seed course's text (item-to-item mode) or a free-text query.
            candidate_title: The recommended course's title (for a readable prompt).
            candidate_text: The recommended course's full text.

        Returns:
            A single short sentence, whitespace-normalized; may be empty if the
            model returned a blank reason.

        Raises:
            LLMUnavailable: If the request fails or the reply is not valid JSON.
        """
        prompt = (
            f"{_EXPLAIN_INSTRUCTION}\n\nQuery: {query}\n\n"
            f"Recommended course — {candidate_title}: {candidate_text}"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": _EXPLAIN_SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": RANDOM_SEED},
        }
        try:
            reply = self._post("/api/generate", payload)
            reason = json.loads(reply["response"])["reason"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailable(f"Ollama explain failed: {exc}") from exc
        return _normalize_text(str(reason))


# --------------------------------------------------------------------------- #
# Tag cache (content-addressed, shared across runs)                           #
# --------------------------------------------------------------------------- #


class _TagCache:
    """Per-text tag cache keyed by ``sha1(model + normalized_text)``.

    A single ``tags.json`` mapping each key to a tag dict, shared across every run
    of one model so any course is enriched at most once, ever.
    """

    def __init__(self, model_key: str) -> None:
        """Open (or initialize) the cache for a given model identity."""
        self._model_key = model_key
        self._path = _LLMCACHE_DIR / _slug(model_key) / "tags.json"
        self._store: dict[str, dict] = {}
        if self._path.exists():
            self._store = json.loads(self._path.read_text())
            logger.info("llmcache[%s]: loaded %d tags", model_key, len(self._store))

    def key(self, normalized_text: str) -> str:
        """Hash ``model_key + normalized_text`` to the cache key (the rule's key)."""
        h = hashlib.sha1()
        h.update(self._model_key.encode())
        h.update(b"\x00")
        h.update(normalized_text.encode())
        return h.hexdigest()

    def get(self, normalized_text: str) -> CourseTags | None:
        """Return cached tags for a text, or ``None`` on a miss."""
        payload = self._store.get(self.key(normalized_text))
        return CourseTags.from_dict(payload) if payload is not None else None

    def put(self, normalized_text: str, tags: CourseTags) -> None:
        """Insert tags for a text into the in-memory store (call :meth:`save`)."""
        self._store[self.key(normalized_text)] = tags.to_dict()

    def save(self) -> None:
        """Persist the store to disk (atomic enough for a single local writer)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._store))

    def __len__(self) -> int:
        return len(self._store)


class _RerankCache:
    """LLM rerank cache keyed by ``sha1(model + query + candidate-ids)``.

    The reranker's only persisted output: one ``reranks.json`` mapping each
    (query, candidate-set) to the model's ranked id order, shared across runs so a
    given rerank is generated at most once. The candidate ids are part of the key
    (in retrieval order), so any change to what the base retrieves invalidates the
    entry rather than silently reusing a stale order.
    """

    def __init__(self, model_key: str) -> None:
        """Open (or initialize) the rerank cache for a given model identity."""
        self._model_key = model_key
        self._path = _LLMCACHE_DIR / _slug(model_key) / "reranks.json"
        self._store: dict[str, list[str]] = {}
        if self._path.exists():
            self._store = json.loads(self._path.read_text())
            logger.info(
                "rerankcache[%s]: loaded %d entries", model_key, len(self._store)
            )

    def key(self, query: str, candidate_ids: list[str]) -> str:
        """Hash ``model_key + normalized query + candidate ids`` to the cache key."""
        h = hashlib.sha1()
        h.update(self._model_key.encode())
        h.update(b"\x00")
        h.update(_normalize_text(query).encode())
        h.update(b"\x00")
        h.update("\x00".join(candidate_ids).encode())
        return h.hexdigest()

    def get(self, key: str) -> list[str] | None:
        """Return a cached ranked id list for a key, or ``None`` on a miss."""
        return self._store.get(key)

    def put(self, key: str, ranked_ids: list[str]) -> None:
        """Insert a ranked id list for a key (call :meth:`save` to persist)."""
        self._store[key] = ranked_ids

    def save(self) -> None:
        """Persist the store to disk (atomic enough for a single local writer)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._store))

    def __len__(self) -> int:
        return len(self._store)


class _ExplanationCache:
    """Why-this-fits cache keyed by ``sha1(model + query + candidate-id)``.

    One ``explanations.json`` mapping each (query, candidate) to the model's
    one-line justification, shared across runs so a given explanation is generated
    at most once. The candidate id (not its text) is part of the key, matching the
    rerank cache — the catalog is static, so the id pins the text.
    """

    def __init__(self, model_key: str) -> None:
        """Open (or initialize) the explanation cache for a given model identity."""
        self._model_key = model_key
        self._path = _LLMCACHE_DIR / _slug(model_key) / "explanations.json"
        self._store: dict[str, str] = {}
        if self._path.exists():
            self._store = json.loads(self._path.read_text())
            logger.info(
                "explaincache[%s]: loaded %d entries", model_key, len(self._store)
            )

    def key(self, query: str, candidate_id: str) -> str:
        """Hash ``model_key + normalized query + candidate id`` to the cache key."""
        h = hashlib.sha1()
        h.update(self._model_key.encode())
        h.update(b"\x00")
        h.update(_normalize_text(query).encode())
        h.update(b"\x00")
        h.update(candidate_id.encode())
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        """Return a cached explanation for a key, or ``None`` on a miss."""
        return self._store.get(key)

    def put(self, key: str, reason: str) -> None:
        """Insert an explanation for a key (call :meth:`save` to persist)."""
        self._store[key] = reason

    def save(self) -> None:
        """Persist the store to disk (atomic enough for a single local writer)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._store))

    def __len__(self) -> int:
        return len(self._store)


def enrich_courses(
    courses: pd.DataFrame,
    course_ids: list[str],
    client: OllamaClient,
    *,
    save_every: int = 50,
) -> int:
    """Extract and cache tags for ``course_ids`` (the slow, explicit LLM pass).

    Skips ids already cached, so the pass is resumable: interrupt it and re-run,
    and only the remainder is generated. Persists periodically so progress
    survives a crash.

    Args:
        courses: Processed catalog indexed by ``course_id`` (needs ``title``,
            ``text``).
        course_ids: The ids to enrich (e.g. the eval-relevant subset).
        client: A reachable :class:`OllamaClient`.
        save_every: Persist the cache every this many fresh extractions.

    Returns:
        The number of courses freshly enriched this call (cache hits excluded).

    Raises:
        LLMUnavailable: If the client is not available.
    """
    if not client.available():
        raise LLMUnavailable(f"Ollama unavailable at {client.host} ({client.model})")
    cache = _TagCache(f"ollama:{client.model}")
    fresh = 0
    for cid in course_ids:
        if cid not in courses.index:
            continue
        text = _normalize_text(courses.loc[cid, "text"])
        if cache.get(text) is not None:
            continue
        title = str(courses.loc[cid, "title"] or cid)
        cache.put(text, client.extract_tags(title, text))
        fresh += 1
        if fresh % save_every == 0:
            cache.save()
            logger.info("enriched %d/%d (%s)", fresh, len(course_ids), client.model)
    cache.save()
    logger.info(
        "enrich_courses: %d freshly enriched (%d cached total)", fresh, len(cache)
    )
    return fresh


# --------------------------------------------------------------------------- #
# The recommender                                                             #
# --------------------------------------------------------------------------- #


class LLMTagRecommender(Recommender):
    """Rank by TF-IDF cosine over LLM-distilled tag profiles (raw-text fallback).

    ``fit`` reads the tag cache only — it never calls the LLM (that is
    :func:`enrich_courses`' job). Courses with cached tags use their tag profile;
    the rest fall back to raw text, so the technique never crashes on a cold
    course, only loses the LLM's distillation for it.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        host: str = OLLAMA_HOST,
        stopwords: bool = True,
        ngram_max: int = 1,
        client: OllamaClient | None = None,
    ) -> None:
        """Configure the rung.

        Args:
            model: Ollama model tag whose cache this rung reads.
            host: Ollama base URL (only used for live query enrichment).
            stopwords: Drop English stop words from the profile TF-IDF.
            ngram_max: Largest n-gram in the profile TF-IDF.
            client: Injected client (defaults to a new :class:`OllamaClient`);
                used only to enrich a free-text query, never in ``fit``.
        """
        if ngram_max < 1:
            raise ValueError("ngram_max must be >= 1")
        self.config = {
            "model": model,
            "stopwords": stopwords,
            "ngram_max": ngram_max,
        }
        self._client = client or OllamaClient(model=model, host=host)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: sp.csr_matrix | None = None
        self._course_ids: list[str] = []
        self._row: dict[str, int] = {}
        self._n_enriched = 0
        self.name = f"llm_tags({_slug(model)})"

    # -- profile construction --------------------------------------------------

    def _build_profiles(self, courses: pd.DataFrame) -> tuple[list[str], int]:
        """Build one profile string per course from cached tags, else raw text.

        Returns:
            ``(profiles, n_enriched)`` — the per-course strings (row-aligned with
            ``courses``) and how many used real LLM tags (the rest fell back to
            raw text). A pathologically empty profile falls back to the id so the
            vectorizer never sees an empty document.
        """
        cache = _TagCache(f"ollama:{self.config['model']}")
        profiles: list[str] = []
        n_enriched = 0
        for cid in courses.index:
            text = _normalize_text(courses.loc[cid, "text"])
            tags = cache.get(text)
            if tags is not None and tags.profile_text():
                profiles.append(tags.profile_text())
                n_enriched += 1
            else:
                profiles.append(text or _normalize_text(cid))
        return profiles, n_enriched

    # -- fit + persistence -----------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Vectorize the tag profiles, loading a cached artifact when one matches.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.

        Raises:
            LLMUnavailable: If no course has cached tags *and* Ollama is
                unreachable — the rung would be an empty-handed raw-text duplicate,
                so it skips with a flag instead (plan §1 graceful degradation).
        """
        self._course_ids = list(courses.index)
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        profiles, n_enriched = self._build_profiles(courses)
        self._n_enriched = n_enriched
        if n_enriched == 0 and not self._client.available():
            raise LLMUnavailable(
                f"{self.name}: no tags cached and Ollama unreachable — run "
                "scripts/enrich_catalog.py with `ollama serve` up first"
            )
        if n_enriched == 0:
            logger.warning(
                "%s: no tags cached — falling back to raw text for all courses; "
                "run scripts/enrich_catalog.py to enrich",
                self.name,
            )

        fingerprint = self._fingerprint(profiles)
        if self._load(fingerprint):
            logger.info("%s: loaded cached artifact", self.name)
            return
        logger.info(
            "%s: vectorizing %d profiles (%d LLM-enriched, %d raw-text fallback)",
            self.name,
            len(profiles),
            n_enriched,
            len(profiles) - n_enriched,
        )
        self._vectorizer = TfidfVectorizer(
            stop_words="english" if self.config["stopwords"] else None,
            ngram_range=(1, self.config["ngram_max"]),
            lowercase=True,
        )
        self._matrix = self._vectorizer.fit_transform(profiles).tocsr()
        self._save(fingerprint)

    def _fingerprint(self, profiles: list[str]) -> str:
        """Hash config + profiles so a stale or re-enriched cache is rejected."""
        h = hashlib.sha1()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(b"\x00".join(p.encode() for p in profiles))
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
        self._n_enriched = int(meta.get("n_enriched", 0))
        return True

    def _save(self, fingerprint: str) -> None:
        """Persist the fitted vectorizer, profile matrix, and fingerprint."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        with (self._artifact_dir / "vectorizer.pkl").open("wb") as fh:
            pickle.dump(self._vectorizer, fh)
        sp.save_npz(self._artifact_dir / "matrix.npz", self._matrix)
        meta = {
            "name": self.name,
            "config": self.config,
            "n_enriched": self._n_enriched,
            "fingerprint": fingerprint,
        }
        (self._artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- recommendation --------------------------------------------------------

    def _rank(self, qvec: sp.spmatrix, k: int, exclude_row: int | None) -> list[Rec]:
        """Score the catalog against a query vector and return the top-k recs."""
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
        """Recommend courses similar to a seed (tag-profile cosine), seed excluded.

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
        row = self._row[course_id]
        return self._rank(self._matrix[row], k, exclude_row=row)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Recommend courses for a free-text query in the same tag-profile space.

        If Ollama is reachable the query is enriched into a tag profile (cached
        like any course); otherwise it falls back to the raw query text. Either
        way it is transformed by the fitted profile vectorizer.

        Args:
            query: A natural-language query.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec`, sorted by descending score.
        """
        if self._vectorizer is None:
            raise RuntimeError(f"{self.name}: fit before recommending")
        profile = self._enrich_query(query)
        return self._rank(self._vectorizer.transform([profile]), k, exclude_row=None)

    def _enrich_query(self, query: str) -> str:
        """Turn a query into a tag profile via the LLM (cached), else raw text."""
        normalized = _normalize_text(query)
        if not normalized:
            return ""
        cache = _TagCache(f"ollama:{self.config['model']}")
        cached = cache.get(normalized)
        if cached is not None:
            return cached.profile_text() or normalized
        if not self._client.available():
            return normalized
        try:
            tags = self._client.extract_tags(normalized, normalized)
        except LLMUnavailable:
            return normalized
        cache.put(normalized, tags)
        cache.save()
        return tags.profile_text() or normalized


# --------------------------------------------------------------------------- #
# Zero-shot LLM reranker (Track B.8b)                                          #
# --------------------------------------------------------------------------- #


class LLMRerankRecommender(Recommender):
    """Retrieve top-N with a base ranker, then reorder them with one LLM call.

    Where :class:`LLMTagRecommender` *distills* each course to tags and ranks in
    that lossy space, this rung keeps a strong first-stage ranker (SBERT by
    default) and uses the LLM only to **reorder** its top-N candidates, reading
    each candidate's *full* text — no distillation, so no signal is thrown away
    before the model sees it. It is the cross-encoder rerank idea (``rerank.py``)
    with a zero-shot LLM in place of a trained cross-encoder (plan §2.8b).

    Why this and not the tag rung
    -----------------------------
    The tag rung settled as not competitive: collapsing a description to ~6–12
    tags loses more than the LLM's normalization adds (ADR-0009). A reranker
    sidesteps that — it never compresses the catalog, only judges a handful of
    already-relevant candidates against the query in one shot.

    Determinism, caching, graceful degradation
    ------------------------------------------
    One deterministic Ollama call (``temperature=0``, ``seed=RANDOM_SEED``) per
    (query, candidate-set), cached by ``sha1(model + query + candidate-ids)`` in
    ``artifacts/llmcache/<model>/reranks.json``. When Ollama is unreachable the
    rung **falls back to the base order** rather than failing — so the worst case
    is "no better than the base," never a crashed suite (plan §1). ``fit`` skips
    (raises :class:`LLMUnavailable`) only in the cold case — Ollama down *and* no
    rerank cached — where every query would reproduce the base and the row would
    be a useless duplicate.
    """

    def __init__(
        self,
        *,
        base: Recommender | None = None,
        model: str = DEFAULT_LLM_MODEL,
        host: str = OLLAMA_HOST,
        retrieve_n: int = 20,
        candidate_chars: int = 1000,
        client: OllamaClient | None = None,
    ) -> None:
        """Configure the retrieve-then-rerank pipeline.

        Args:
            base: First-stage retriever (any fitted-capable :class:`Recommender`).
                Defaults to a MiniLM :class:`SbertRecommender`.
            model: Ollama model tag used to rerank (and to key the cache).
            host: Ollama base URL (only used for live reranking).
            retrieve_n: How many candidates the base retrieves before reranking.
            candidate_chars: Truncate each candidate's text to this many chars in
                the prompt, bounding prompt size on long descriptions.
            client: Injected client (defaults to a new :class:`OllamaClient`).

        Raises:
            ValueError: If ``retrieve_n`` or ``candidate_chars`` is not positive.
        """
        if retrieve_n < 1:
            raise ValueError("retrieve_n must be >= 1")
        if candidate_chars < 1:
            raise ValueError("candidate_chars must be >= 1")
        self._base = base if base is not None else SbertRecommender()
        self._client = client or OllamaClient(model=model, host=host)
        self._text_by_id: dict[str, str] = {}
        self._cache: _RerankCache | None = None
        self.config = {
            "base": self._base.name,
            "model": model,
            "retrieve_n": retrieve_n,
            "candidate_chars": candidate_chars,
        }
        self.name = (
            f"llm_rerank({_slug(model)},base={_slug(self._base.name)},n={retrieve_n})"
        )

    # -- fit ------------------------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Fit the base retriever and capture candidate texts for reranking.

        The base persists/reloads its own artifact; this stage adds only the
        per-query rerank cache (filled lazily at query time). No precomputed
        rerank artifact exists — reranking is query-dependent.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id``.

        Raises:
            LLMUnavailable: If Ollama is unreachable *and* no rerank is cached —
                the rung would reproduce the base order for every query and be a
                useless duplicate, so it skips with a flag instead (plan §1).
        """
        self._cache = _RerankCache(f"ollama:{self.config['model']}")
        if not self._client.available() and len(self._cache) == 0:
            raise LLMUnavailable(
                f"{self.name}: Ollama unreachable and no rerank cached — start "
                "`ollama serve` so the reranker can reorder candidates"
            )
        self._base.fit(courses)
        text = courses["text"].fillna("")
        self._text_by_id = {
            cid: _normalize_text(t) or _normalize_text(cid)
            for cid, t in zip(courses.index, text, strict=True)
        }

    # -- reranking core -------------------------------------------------------

    def _reconcile(self, order: list[str], candidates: list[str]) -> list[str]:
        """Force ``order`` to a full permutation of ``candidates``.

        Drops ids the model invented or duplicated and appends any candidate it
        omitted, in base order — so the result always ranks every candidate
        exactly once, however the model (mis)behaved.
        """
        candset = set(candidates)
        seen: set[str] = set()
        reconciled = [
            c for c in order if c in candset and not (c in seen or seen.add(c))
        ]
        reconciled.extend(c for c in candidates if c not in seen)
        return reconciled

    def _live_order(self, query: str, candidates: list[str]) -> list[str]:
        """Ask the LLM to reorder ``candidates``; fall back to base order on failure."""
        texts = [
            self._text_by_id[c][: self.config["candidate_chars"]] for c in candidates
        ]
        try:
            idx = self._client.rank_candidates(query, texts)
        except LLMUnavailable:
            logger.warning("%s: rerank failed — using base order", self.name)
            return list(candidates)
        order = [candidates[i] for i in idx if 0 <= i < len(candidates)]
        return self._reconcile(order, candidates)

    def _ranked_order(self, query: str, candidates: list[str]) -> list[str]:
        """Return the reranked candidate ids: cache, else live LLM, else base order."""
        assert self._cache is not None  # set in fit()
        key = self._cache.key(query, candidates)
        cached = self._cache.get(key)
        if cached is not None:
            return self._reconcile(cached, candidates)
        if not self._client.available():
            return list(candidates)
        order = self._live_order(query, candidates)
        self._cache.put(key, order)
        self._cache.save()
        return order

    def _rerank(self, query: str, candidates: list[str], k: int) -> list[Rec]:
        """Rerank candidates for a query and return the top-k as rank-scored recs.

        Scores are rank-based (``len - position``), strictly descending, so the
        list honors the interface contract regardless of the base's own scores.
        """
        if not candidates:
            return []
        order = self._ranked_order(query, candidates)[:k]
        n = len(order)
        return [Rec(cid, float(n - i)) for i, cid in enumerate(order)]

    # -- recommendation -------------------------------------------------------

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses similar to a seed: base retrieval, LLM-reranked.

        The base retrieves ``retrieve_n`` candidates (seed already excluded); the
        LLM reorders them against the seed's own text.

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
        """Recommend courses for a free-text query: base retrieval, LLM-reranked.

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


# --------------------------------------------------------------------------- #
# "Why this fits" explainer (Track B.8c)                                       #
# --------------------------------------------------------------------------- #


class RecommendationExplainer:
    """Generate a one-line "why this fits" justification for a recommendation.

    This is the last Track B.8 piece (plan §2.8c) and the one place the evidence
    says the local LLM earns its cost: not *ranking* (the tag rung and zero-shot
    reranker both lost to the SBERT base — ADR-0009, ADR-0010), but *explaining* a
    ranking already produced by a stronger method. Given a query (a seed course's
    text or a free-text search) and one already-recommended candidate, it returns
    a short sentence naming the shared topic/skill, for the Phase 8 UI's "why this
    fits" line.

    Not a ranker, by design
    -----------------------
    It is **not** a :class:`Recommender` — it produces no ordering and is never
    scored by the eval harness or the leaderboard (an explanation has no
    ground-truth ranking to measure). It is a presentation helper layered on top
    of whatever rung the UI is showing (ADR-0011).

    Determinism, caching, graceful degradation
    ------------------------------------------
    One deterministic Ollama call (``temperature=0``, ``seed=RANDOM_SEED``) per
    (query, candidate), cached by ``sha1(model + query + candidate-id)`` in
    ``artifacts/llmcache/<model>/explanations.json``. Because the "why" line is an
    optional UI nicety, every unavailability path degrades to ``None`` (the UI
    simply omits the line) rather than raising — so ``fit`` never skips and a cold,
    offline call is a quiet no-op, not a crash (plan §1).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        host: str = OLLAMA_HOST,
        candidate_chars: int = 1000,
        client: OllamaClient | None = None,
    ) -> None:
        """Configure the explainer.

        Args:
            model: Ollama model tag used to explain (and to key the cache).
            host: Ollama base URL (only used for live explanation).
            candidate_chars: Truncate the query text and the candidate text to this
                many chars in the prompt, bounding prompt size on long descriptions.
            client: Injected client (defaults to a new :class:`OllamaClient`).

        Raises:
            ValueError: If ``candidate_chars`` is not positive.
        """
        if candidate_chars < 1:
            raise ValueError("candidate_chars must be >= 1")
        self._client = client or OllamaClient(model=model, host=host)
        self._text_by_id: dict[str, str] = {}
        self._title_by_id: dict[str, str] = {}
        self._cache: _ExplanationCache | None = None
        self.config = {"model": model, "candidate_chars": candidate_chars}
        self.name = f"llm_explain({_slug(model)})"

    def fit(self, courses: pd.DataFrame) -> RecommendationExplainer:
        """Capture candidate texts/titles and open the explanation cache.

        No LLM call happens here — generation is lazy, at :meth:`explain` time.

        Args:
            courses: Processed catalog (one row per course), indexed by
                ``course_id`` with ``title`` and ``text`` columns.

        Returns:
            ``self``, so callers can ``RecommendationExplainer(...).fit(courses)``.
        """
        self._cache = _ExplanationCache(f"ollama:{self.config['model']}")
        text = courses["text"].fillna("")
        self._text_by_id = {
            cid: _normalize_text(t) or _normalize_text(cid)
            for cid, t in zip(courses.index, text, strict=True)
        }
        self._title_by_id = {
            cid: str(courses.loc[cid, "title"] or cid) for cid in courses.index
        }
        return self

    def explain(self, query: str, candidate_id: str) -> str | None:
        """Explain why ``candidate_id`` fits ``query`` (cache, else live, else None).

        Args:
            query: The seed course's text (item-to-item mode) or a free-text query.
            candidate_id: An already-recommended course to justify; must exist in
                the fitted catalog.

        Returns:
            A one-line justification, or ``None`` when the query is empty, the
            model returns nothing, or Ollama is unreachable with no cached
            explanation (the UI then omits the line).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            KeyError: If ``candidate_id`` is not in the fitted catalog.
        """
        if self._cache is None:
            raise RuntimeError(f"{self.name}: fit before explaining")
        if candidate_id not in self._text_by_id:
            raise KeyError(f"unknown course_id: {candidate_id!r}")
        normalized = _normalize_text(query)
        if not normalized:
            return None
        key = self._cache.key(normalized, candidate_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not self._client.available():
            return None
        chars = self.config["candidate_chars"]
        try:
            reason = self._client.explain(
                normalized[:chars],
                self._title_by_id[candidate_id],
                self._text_by_id[candidate_id][:chars],
            )
        except LLMUnavailable:
            logger.warning("%s: explain failed — omitting the line", self.name)
            return None
        if not reason:
            return None
        self._cache.put(key, reason)
        self._cache.save()
        return reason

    def explain_seed(self, seed_id: str, candidate_id: str) -> str | None:
        """Explain a similar-course recommendation using the seed's own text.

        Convenience wrapper for item-to-item mode: resolves ``seed_id`` to its
        text and explains ``candidate_id`` against it.

        Args:
            seed_id: The seed course the recommendation was made from.
            candidate_id: The recommended course to justify.

        Returns:
            A one-line justification, or ``None`` (see :meth:`explain`).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            KeyError: If either id is not in the fitted catalog.
        """
        if seed_id not in self._text_by_id:
            raise KeyError(f"unknown course_id: {seed_id!r}")
        return self.explain(self._text_by_id[seed_id], candidate_id)
