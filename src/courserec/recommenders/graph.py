"""Course graph + personalized PageRank (random walk with restart).

The technique builds a graph over the catalog and ranks recommendations by graph
*proximity* to the seed (recommender_plan.md §2.6 / Track B.6). It is the **one
technique permitted to read** ``Cross-Listed Course(s)`` as an input feature —
every other method treats that column as the evaluation target. To keep the
read leakage-free, the graph is fit only on a *training* subset of cross-listing
edges and evaluated only on the *held-out* edges it never saw (see
:func:`courserec.eval.split_crosslist_edges`).

The graph
---------
Nodes are courses plus two kinds of lightweight **auxiliary nodes** — one per
subject and one per department. Each course attaches to its subject node and its
department node, so two courses in the same subject are two hops apart *through*
the subject node rather than via a dense same-subject clique (242 subjects would
otherwise add millions of edges). Cross-listed courses are joined directly by a
high-weight edge. Edge weights:

* ``w_xlist`` — direct cross-listing edge (the strong, sparse signal).
* ``w_meta``  — course→subject and course→department edges (weak, dense glue).

Math sketch
-----------
Let ``A`` be the symmetric weighted adjacency and ``P = A D⁻¹`` the
column-stochastic transition matrix (``D`` the diagonal degree matrix). Random
walk with restart from seed ``s`` is the fixed point

    r = (1 − c)·P·r + c·eₛ

where ``c`` is the restart probability and ``eₛ`` the indicator of the seed.
``r`` is the stationary visit distribution of a walker that, at every step,
teleports back to ``s`` with probability ``c``. We solve it by power iteration
(a handful of sparse mat-vecs). Recommendations are the highest-scoring *course*
nodes (auxiliary nodes and the seed are dropped).

Complexity
----------
Fit is ``O(E)`` to assemble a sparse graph with ``E ≈ n_courses·2 + n_xlist``
edges (tens of thousands). A query is ``O(n_iter · nnz)`` — a few dozen sparse
mat-vecs.

Persisted artifacts
-------------------
``artifacts/<slug>/`` holds the adjacency matrix (``graph.npz``), the node index
(``nodes.json``), and a ``meta.json`` fingerprint over the config + node/edge
structure so a stale or differently-split graph is rejected.

When it wins / loses
--------------------
Wins when a held-out twin is reachable through *remaining* structure — a course
cross-listed three-ways still links to a twin after one edge is removed, and
metadata glue surfaces same-subject neighbors. Loses on isolated pairs: if the
only edge between two twins is the held-out one and they share no subject/dept,
no amount of walking can recover it. That honest ceiling is the point of the
held-out evaluation. The method is item-to-item only — it has no text encoder,
so ``recommend_by_text`` raises :class:`NotImplementedError`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from courserec.config import ARTIFACTS_DIR
from courserec.eval import crosslist_edges
from courserec.interfaces import Rec, Recommender

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """Turn a technique ``name`` into a filesystem-safe artifact directory slug."""
    return _SLUG_RE.sub("_", name.lower()).strip("_")


class GraphRecommender(Recommender):
    """Rank by personalized-PageRank proximity in a course/metadata graph.

    The sole technique allowed to use cross-listings as a feature; it must be
    evaluated on a held-out edge split (see the module docstring and
    :func:`courserec.eval.split_crosslist_edges`).
    """

    def __init__(
        self,
        *,
        use_metadata: bool = True,
        w_xlist: float = 1.0,
        w_meta: float = 0.3,
        restart: float = 0.15,
        n_iter: int = 60,
        tol: float = 1e-6,
        held_out_edges: frozenset[frozenset[str]] | None = None,
    ) -> None:
        """Configure the graph and the random walk.

        Args:
            use_metadata: If true, add course→subject and course→department
                auxiliary nodes (the dense "glue"). If false, the graph contains
                only cross-listing edges — a pure structural baseline.
            w_xlist: Weight of a direct cross-listing edge.
            w_meta: Weight of a course→subject / course→department edge.
            restart: Random-walk restart (teleport) probability ``c`` in
                ``(0, 1)``; higher keeps the walk closer to the seed.
            n_iter: Maximum power-iteration steps per query.
            tol: L1 convergence tolerance that stops the power iteration early.
            held_out_edges: Cross-listing edges to **exclude** from the graph —
                the evaluation target, withheld to keep the read leakage-free.
                ``None`` builds the full graph (use only for ad-hoc inspection,
                never for the held-out leaderboard).

        Raises:
            ValueError: If ``restart`` is not in ``(0, 1)`` or weights are
                non-positive.
        """
        if not 0.0 < restart < 1.0:
            raise ValueError("restart must be in (0, 1)")
        if w_xlist <= 0 or w_meta <= 0:
            raise ValueError("edge weights must be positive")
        self._held_out = frozenset(held_out_edges or frozenset())
        self.config = {
            "use_metadata": use_metadata,
            "w_xlist": w_xlist,
            "w_meta": w_meta,
            "restart": restart,
            "n_iter": n_iter,
            "tol": tol,
            "n_held_out": len(self._held_out),
        }
        self.name = (
            f"graph(meta={'on' if use_metadata else 'off'},"
            f"wx={w_xlist},wm={w_meta},c={restart})"
        )
        self._adj: sp.csr_matrix | None = None  # symmetric weighted adjacency
        self._inv_deg: np.ndarray | None = None  # 1 / degree, for the transition
        self._course_ids: list[str] = []
        self._row: dict[str, int] = {}
        self._n_courses = 0

    # -- fit + persistence -----------------------------------------------------

    def fit(self, courses: pd.DataFrame) -> None:
        """Build (or load) the graph, excluding any held-out cross-listing edges.

        Args:
            courses: Processed catalog indexed by ``course_id`` with ``subject``
                and ``department`` columns.
        """
        self._course_ids = list(courses.index)
        self._row = {cid: i for i, cid in enumerate(self._course_ids)}
        self._n_courses = len(self._course_ids)

        edges = crosslist_edges(courses)
        train_edges = edges - self._held_out  # the sanctioned, leakage-free read
        fingerprint = self._fingerprint(courses, train_edges)
        if self._load(fingerprint):
            logger.info("%s: loaded cached graph", self.name)
            return

        logger.info(
            "%s: building graph (%d courses, %d cross-list edges, %d held out)",
            self.name,
            self._n_courses,
            len(train_edges),
            len(self._held_out),
        )
        self._adj = self._build_adjacency(courses, train_edges)
        self._finalize_transition()
        self._save(fingerprint)

    def _build_adjacency(
        self, courses: pd.DataFrame, train_edges: set[frozenset[str]]
    ) -> sp.csr_matrix:
        """Assemble the symmetric weighted adjacency over course + aux nodes."""
        rows, cols, data = [], [], []

        def add(i: int, j: int, w: float) -> None:
            """Add an undirected edge (both directions) of weight ``w``."""
            rows.extend((i, j))
            cols.extend((j, i))
            data.extend((w, w))

        for edge in train_edges:
            a, b = tuple(edge)
            add(self._row[a], self._row[b], self.config["w_xlist"])

        n_nodes = self._n_courses
        if self.config["use_metadata"]:
            # Auxiliary nodes are appended after the course block, so course rows
            # keep indices 0..n_courses-1 and stay sliceable at query time.
            n_nodes = self._add_metadata_nodes(courses, rows, cols, data)

        adj = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float64
        )
        adj.sum_duplicates()
        return adj

    def _add_metadata_nodes(
        self,
        courses: pd.DataFrame,
        rows: list[int],
        cols: list[int],
        data: list[float],
    ) -> int:
        """Append subject/department aux nodes and their edges; return node count.

        Each distinct subject and department becomes one node that every course
        in it links to — a star, not a clique, so the graph stays sparse.
        """
        w = self.config["w_meta"]
        next_id = self._n_courses
        group_node: dict[tuple[str, str], int] = {}
        for col in ("subject", "department"):
            values = courses[col]
            for cid, value in values.items():
                if not isinstance(value, str) or not value:
                    continue  # missing metadata: simply no edge for that facet
                key = (col, value)
                node = group_node.get(key)
                if node is None:
                    node = next_id
                    group_node[key] = node
                    next_id += 1
                rows.extend((self._row[cid], node))
                cols.extend((node, self._row[cid]))
                data.extend((w, w))
        logger.info("%s: %d metadata aux nodes", self.name, len(group_node))
        return next_id

    def _finalize_transition(self) -> None:
        """Precompute inverse degrees for the column-stochastic transition."""
        assert self._adj is not None
        deg = np.asarray(self._adj.sum(axis=1)).ravel()
        # Isolated nodes (degree 0) get inv_deg 0; the restart term still keeps
        # the walk well-defined, and such a seed simply yields no recommendations.
        with np.errstate(divide="ignore"):
            inv = np.where(deg > 0, 1.0 / deg, 0.0)
        self._inv_deg = inv

    def _fingerprint(
        self, courses: pd.DataFrame, train_edges: set[frozenset[str]]
    ) -> str:
        """Hash config + nodes + edges so a stale/mismatched cache is rejected."""
        h = hashlib.sha1()
        h.update(json.dumps(self.config, sort_keys=True).encode())
        h.update(b"\x00".join(c.encode() for c in self._course_ids))
        if self.config["use_metadata"]:
            for col in ("subject", "department"):
                h.update(col.encode())
                h.update(b"\x00".join(str(v).encode() for v in courses[col]))
        edge_keys = sorted(tuple(sorted(e)) for e in train_edges)
        h.update(b"\x00".join(f"{a}|{b}".encode() for a, b in edge_keys))
        return h.hexdigest()

    @property
    def _artifact_dir(self):
        return ARTIFACTS_DIR / _slug(self.name)

    def _load(self, fingerprint: str) -> bool:
        """Load a cached adjacency + node index if the fingerprint matches."""
        meta_path = self._artifact_dir / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            return False
        self._adj = sp.load_npz(self._artifact_dir / "graph.npz").tocsr()
        self._n_courses = meta["n_courses"]
        self._finalize_transition()
        return True

    def _save(self, fingerprint: str) -> None:
        """Persist the adjacency matrix, node index, and fingerprint."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(self._artifact_dir / "graph.npz", self._adj)
        (self._artifact_dir / "nodes.json").write_text(json.dumps(self._course_ids))
        meta = {
            "name": self.name,
            "config": self.config,
            "fingerprint": fingerprint,
            "n_courses": self._n_courses,
            "n_nodes": int(self._adj.shape[0]),
        }
        (self._artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- random walk with restart ----------------------------------------------

    def _rwr(self, seed_row: int) -> np.ndarray:
        """Run random walk with restart from one seed; return per-node scores.

        Power-iterates ``r = (1 − c)·A·(r ⊙ inv_deg) + c·eₛ`` until the L1 change
        falls below ``tol`` or ``n_iter`` steps elapse. The ``A·(r ⊙ inv_deg)``
        form applies the column-stochastic transition without materializing it.
        """
        assert self._adj is not None and self._inv_deg is not None
        c = self.config["restart"]
        n = self._adj.shape[0]
        restart = np.zeros(n)
        restart[seed_row] = 1.0
        r = restart.copy()
        for _ in range(self.config["n_iter"]):
            nxt = (1.0 - c) * (self._adj @ (r * self._inv_deg)) + c * restart
            if np.abs(nxt - r).sum() < self.config["tol"]:
                r = nxt
                break
            r = nxt
        return r

    def _rank(self, scores: np.ndarray, k: int, exclude_row: int) -> list[Rec]:
        """Return the top-``k`` course nodes by score, excluding the seed."""
        course_scores = scores[: self._n_courses].copy()
        course_scores[exclude_row] = -np.inf
        n_pos = int((course_scores > 0).sum())
        if n_pos == 0:
            return []
        top = min(k, n_pos)
        idx = np.argpartition(course_scores, -top)[-top:]
        idx = idx[np.argsort(course_scores[idx])[::-1]]
        return [Rec(self._course_ids[i], float(course_scores[i])) for i in idx]

    def recommend_similar(self, course_id: str, k: int = 10) -> list[Rec]:
        """Recommend courses by graph proximity, excluding the seed itself.

        Args:
            course_id: The seed course's id.
            k: Maximum number of recommendations.

        Returns:
            Up to ``k`` :class:`Rec` sorted by descending proximity score. An
            isolated seed (no edges left after the held-out split) yields ``[]``.

        Raises:
            KeyError: If ``course_id`` is not in the fitted catalog.
        """
        if course_id not in self._row:
            raise KeyError(f"unknown course_id: {course_id!r}")
        row = self._row[course_id]
        return self._rank(self._rwr(row), k, exclude_row=row)

    def recommend_by_text(self, query: str, k: int = 10) -> list[Rec]:
        """Not supported: the graph has no text encoder (item-to-item only).

        Raises:
            NotImplementedError: Always — the free-text lens skips this technique.
        """
        raise NotImplementedError(
            "GraphRecommender is item-to-item only; it has no text encoder."
        )
