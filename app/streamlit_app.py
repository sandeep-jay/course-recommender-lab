"""Phase 8 minimal Streamlit UI for the course recommender lab (plan §4).

Three views, no auth / database / styling beyond Streamlit defaults:

* **Explore** — pick a course (or type a free-text query) and a technique, see the
  top-k with scores and the optional "why this fits" line (Phase 7c explainer).
* **Compare** — the same query, two techniques side by side.
* **Leaderboard** — `results/leaderboard.csv` as a sortable table, plus the Phase 6
  UMAP map.

Run it (after `pip install -e ".[ui,semantic]"`) with::

    streamlit run app/streamlit_app.py

Every fitted technique loads from `artifacts/` if present, so the first interaction
is fast; the explainer needs a running Ollama daemon and silently omits its line
when one is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts this file's own dir (app/) on sys.path,
# not the repo root, so the `app` package isn't importable. Put the repo root first
# (pytest does the equivalent via `pythonpath` in pyproject).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.glossary import (  # noqa: E402
    FAMILIES,
    FAMILY_DESCRIPTIONS,
    LEAKAGE_NOTE,
    LENSES,
    TECHNIQUE_INFO,
    family_label,
    metric_help,
)
from app.projection import load_or_compute_coords  # noqa: E402
from app.registry import (  # noqa: E402
    DEFAULT_TECHNIQUE,
    course_label,
    make_recommender,
    technique_names,
)
from courserec.cluster import load_sbert_embeddings  # noqa: E402
from courserec.config import PLOTS_DIR, RANDOM_SEED, RESULTS_DIR  # noqa: E402
from courserec.data import load_processed  # noqa: E402
from courserec.interfaces import Rec, Recommender  # noqa: E402
from courserec.recommenders.embeddings import EmbeddingsUnavailable  # noqa: E402
from courserec.recommenders.llm import RecommendationExplainer  # noqa: E402

LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"
EMBEDDING_MAP_PNG = PLOTS_DIR / "embedding_map.png"
_MAX_K = 20
# The Map view projects this model's embeddings (the default rung) and overlays its
# recommendations, so the points and the highlight come from the same space.
_MAP_MODEL = "all-MiniLM-L6-v2"
_MAP_HEIGHT = 540  # px; the interactive scatter's plotting height.
_MAP_TOP_SUBJECTS = 12  # color the N most common subjects, the rest as "(other)".

# Vega refuses >5000 rows by default; the catalog is ~11k points, all sent to render.
alt.data_transformers.disable_max_rows()


# --- cached resources -------------------------------------------------------
# st.cache_resource keeps one fitted instance per key across reruns, so picking a
# course or flipping a toggle never re-fits or re-reads the catalog.


@st.cache_resource(show_spinner="Loading catalog…")
def _courses() -> pd.DataFrame:
    """Load the processed catalog once per session."""
    return load_processed()


@st.cache_resource(show_spinner=False)
def _course_labels() -> list[str]:
    """Build the seed-course picker labels (``"<id> — <title>"``), id-sorted."""
    courses = _courses()
    return [course_label(cid, courses.loc[cid, "title"]) for cid in courses.index]


@st.cache_resource(show_spinner="Fitting technique…")
def _fitted(name: str) -> Recommender:
    """Instantiate and fit the technique ``name`` (cached per technique)."""
    rec = make_recommender(name)
    rec.fit(_courses())
    return rec


@st.cache_resource(show_spinner=False)
def _explainer() -> RecommendationExplainer:
    """Fit the "why this fits" explainer once (lazy, degrades to None offline)."""
    return RecommendationExplainer().fit(_courses())


@st.cache_resource(show_spinner="Projecting the catalog to 2-D…")
def _map_frame(method: str) -> tuple[pd.DataFrame, str]:
    """Project the SBERT catalog to 2-D (cached) and join course metadata.

    Args:
        method: ``"auto"`` (UMAP if installed, else t-SNE) or ``"tsne"``.

    Returns:
        A ``(DataFrame, used_method)`` pair: one row per course with ``x``/``y``
        coordinates, ``course_id``, ``title``, ``subject``; and the projector used.
    """
    courses = _courses()
    embeddings, course_ids = load_sbert_embeddings(courses, model_name=_MAP_MODEL)
    coords, used = load_or_compute_coords(
        embeddings, method, model_name=_MAP_MODEL, seed=RANDOM_SEED
    )
    meta = courses.loc[course_ids]
    frame = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "course_id": course_ids,
            "title": meta["title"].to_numpy(),
            "subject": meta["subject"].to_numpy(),
        }
    )
    return frame, used


# --- shared helpers ---------------------------------------------------------


def _label_to_id(label: str) -> str:
    """Recover a course id from a ``"<id> — <title>"`` picker label."""
    return label.split(" — ", 1)[0]


def _recommend(
    rec: Recommender, *, seed_id: str | None, query: str, k: int
) -> list[Rec]:
    """Dispatch to item-to-item or free-text recommendation for one technique.

    Returns an empty list (and surfaces a Streamlit message) when a technique is
    text-incapable in query mode, rather than crashing the view.
    """
    try:
        if seed_id is not None:
            return rec.recommend_similar(seed_id, k=k)
        return rec.recommend_by_text(query, k=k)
    except NotImplementedError:
        st.info(f"**{rec.name}** does not support free-text queries.")
        return []


def _results_frame(
    recs: list[Rec],
    courses: pd.DataFrame,
    *,
    explain=None,
) -> pd.DataFrame:
    """Build a display table from recommendations, optionally with a "why" column."""
    rows = []
    for r in recs:
        row = {
            "course_id": r.course_id,
            "title": courses.loc[r.course_id, "title"],
            "subject": courses.loc[r.course_id, "subject"],
            "score": round(r.score, 4),
        }
        if explain is not None:
            row["why this fits"] = explain(r.course_id) or "—"
        rows.append(row)
    return pd.DataFrame(rows)


def _query_controls(key: str) -> tuple[str | None, str]:
    """Render the mode toggle + seed/query input; return ``(seed_id, query)``.

    Exactly one of the two is meaningful: in "Similar to a course" mode ``seed_id``
    is set and ``query`` is the seed's id-label; in "Free-text query" mode
    ``seed_id`` is ``None``.
    """
    mode = st.radio(
        "Query mode",
        ["Similar to a course", "Free-text query"],
        horizontal=True,
        key=f"{key}_mode",
    )
    if mode == "Similar to a course":
        label = st.selectbox("Seed course", _course_labels(), key=f"{key}_seed")
        seed_id = _label_to_id(label)
        return seed_id, seed_id
    query = st.text_input(
        "Search query",
        value="practical deep learning",
        key=f"{key}_query",
    )
    return None, query


def _technique_blurb(name: str) -> None:
    """Render the one-line plain-language description for a technique, if any."""
    info = TECHNIQUE_INFO.get(name)
    if info:
        st.caption(info)


def _column_config(columns) -> dict:
    """Build per-column header tooltips for the leaderboard table.

    Args:
        columns: The leaderboard DataFrame's column names.

    Returns:
        A Streamlit ``column_config`` mapping each known metric column to a help
        tooltip (hover the header); columns with no glossary entry are omitted.
    """
    config = {
        col: st.column_config.Column(help=help_text)
        for col in columns
        if (help_text := metric_help(col)) is not None
    }
    config["family"] = st.column_config.Column(
        help="Technique family — definitions in “Technique families” below."
    )
    return config


def _leaderboard_glossary(columns) -> None:
    """Render the lenses / metrics / families explainers below the table."""
    with st.expander("How to read this leaderboard"):
        st.markdown("**Three evaluation lenses** — no single one is sufficient:")
        for title, desc in LENSES:
            st.markdown(f"- **{title}.** {desc}")
        st.caption(LEAKAGE_NOTE)
        st.markdown("**Metrics** (the same text appears on each column's header):")
        for col in columns:
            help_text = metric_help(col)
            if help_text and col != "config":
                st.markdown(f"- **`{col}`** — {help_text}")
    with st.expander("Technique families"):
        for key, label in FAMILIES.items():
            if key != "other":
                st.markdown(f"- **{label}** — {FAMILY_DESCRIPTIONS[key]}")


_MAP_X = alt.X("x:Q", axis=None, scale=alt.Scale(zero=False))
_MAP_Y = alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False))
_MAP_TOOLTIP = ["course_id", "title", "subject"]


def _map_base_chart(frame: pd.DataFrame) -> alt.Chart:
    """Scatter every course, colored by its (top-N) subject — the plain explore map."""
    top = frame["subject"].value_counts().head(_MAP_TOP_SUBJECTS).index
    data = frame.assign(
        group=frame["subject"].where(frame["subject"].isin(top), "(other)")
    )
    return (
        alt.Chart(data)
        .mark_circle(size=18, opacity=0.55)
        .encode(
            x=_MAP_X,
            y=_MAP_Y,
            color=alt.Color("group:N", title=f"Subject (top {_MAP_TOP_SUBJECTS})"),
            tooltip=_MAP_TOOLTIP,
        )
        .properties(height=_MAP_HEIGHT)
        .interactive()
    )


def _map_overlay_chart(
    frame: pd.DataFrame, seed_id: str, rec_ids: set[str]
) -> alt.Chart:
    """Grey the catalog, then light up the seed (diamond) and its recs (triangles)."""
    role = np.where(
        frame["course_id"] == seed_id,
        "seed",
        np.where(frame["course_id"].isin(rec_ids), "recommendation", "other"),
    )
    data = frame.assign(role=role)
    background = (
        alt.Chart(data[data["role"] == "other"])
        .mark_circle(size=12, opacity=0.18, color="lightgray")
        .encode(x=_MAP_X, y=_MAP_Y, tooltip=_MAP_TOOLTIP)
    )
    recs = (
        alt.Chart(data[data["role"] == "recommendation"])
        .mark_point(size=130, shape="triangle-up", filled=True, color="#ff7f0e")
        .encode(x=_MAP_X, y=_MAP_Y, tooltip=_MAP_TOOLTIP)
    )
    seed = (
        alt.Chart(data[data["role"] == "seed"])
        .mark_point(size=360, shape="diamond", filled=True, color="#d62728")
        .encode(x=_MAP_X, y=_MAP_Y, tooltip=_MAP_TOOLTIP)
    )
    return (background + recs + seed).properties(height=_MAP_HEIGHT).interactive()


# --- views ------------------------------------------------------------------


def _view_explore() -> None:
    """Explore: one technique's top-k with scores and the optional why-line."""
    st.subheader("Explore")
    st.caption("Pick a course or type a query, choose a technique, see the top-k.")
    seed_id, query = _query_controls("explore")
    col_t, col_k = st.columns([3, 1])
    names = technique_names()
    technique = col_t.selectbox(
        "Technique", names, index=names.index(DEFAULT_TECHNIQUE), key="explore_tech"
    )
    k = col_k.slider("k", 1, _MAX_K, 10, key="explore_k")
    _technique_blurb(technique)
    want_why = st.checkbox(
        "Explain why each fits (local LLM — needs Ollama)", key="explore_why"
    )

    if seed_id is None and not query.strip():
        st.info("Enter a query to see recommendations.")
        return

    rec = _fitted(technique)
    recs = _recommend(rec, seed_id=seed_id, query=query, k=k)
    if not recs:
        return

    explain = None
    if want_why:
        explainer = _explainer()
        if seed_id is not None:
            explain = lambda cid: explainer.explain_seed(seed_id, cid)  # noqa: E731
        else:
            explain = lambda cid: explainer.explain(query, cid)  # noqa: E731

    courses = _courses()
    with st.spinner("Generating explanations…" if want_why else ""):
        frame = _results_frame(recs, courses, explain=explain)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if want_why and (frame["why this fits"] == "—").all():
        st.caption(
            "No explanations available — start Ollama (`ollama serve`) and pull the "
            "model to populate the *why this fits* line."
        )


def _view_compare() -> None:
    """Compare: the same query under two techniques, side by side."""
    st.subheader("Compare")
    st.caption("Run one query through two techniques and compare the rankings.")
    seed_id, query = _query_controls("compare")
    names = technique_names()
    col_a, col_b = st.columns(2)
    tech_a = col_a.selectbox("Technique A", names, index=0, key="compare_a")
    with col_a:
        _technique_blurb(tech_a)
    tech_b = col_b.selectbox(
        "Technique B", names, index=min(1, len(names) - 1), key="compare_b"
    )
    with col_b:
        _technique_blurb(tech_b)
    k = st.slider("k", 1, _MAX_K, 10, key="compare_k")

    if seed_id is None and not query.strip():
        st.info("Enter a query to compare.")
        return

    courses = _courses()
    for col, tech in ((col_a, tech_a), (col_b, tech_b)):
        recs = _recommend(_fitted(tech), seed_id=seed_id, query=query, k=k)
        with col:
            st.markdown(f"**{tech}**")
            if recs:
                st.dataframe(
                    _results_frame(recs, courses),
                    use_container_width=True,
                    hide_index=True,
                )


def _view_leaderboard() -> None:
    """Leaderboard: the eval table (sortable) plus the UMAP map."""
    st.subheader("Leaderboard")
    if LEADERBOARD_CSV.exists():
        board = pd.read_csv(LEADERBOARD_CSV)
        board.insert(1, "family", board["name"].map(family_label))
        st.caption(
            f"{len(board)} technique×config rows from `{LEADERBOARD_CSV.name}`, "
            "sorted by NDCG@10 — click any column header to re-sort, hover a header "
            "for its definition. Regenerate with `python scripts/run_eval.py`."
        )
        st.dataframe(
            board,
            use_container_width=True,
            hide_index=True,
            column_config=_column_config(board.columns),
        )
        _leaderboard_glossary(board.columns)
    else:
        st.warning(
            "No leaderboard yet — run `python scripts/run_eval.py` to generate "
            f"`{LEADERBOARD_CSV}`."
        )

    st.markdown("#### Catalog map (UMAP)")
    if EMBEDDING_MAP_PNG.exists():
        st.image(
            str(EMBEDDING_MAP_PNG),
            caption="Phase 6 SBERT embeddings projected to 2-D (UMAP).",
            use_container_width=True,
        )
    else:
        st.info("No map yet — run `python scripts/run_clustering.py` to generate it.")
    st.caption("For a live, zoomable version, see the **Map** view.")


def _view_map() -> None:
    """Map: a live, interactive 2-D projection that can highlight a seed + its recs."""
    st.subheader("Map")
    st.caption(
        "Each point is a course, placed by its SBERT embedding projected to 2-D. "
        "Hover for the course, scroll to zoom, drag to pan."
    )
    method_label = st.radio(
        "Projection", ["UMAP (fast)", "t-SNE (slower)"], horizontal=True, key="map_proj"
    )
    method = "tsne" if method_label.startswith("t-SNE") else "auto"
    try:
        frame, used = _map_frame(method)
    except EmbeddingsUnavailable:
        st.warning(
            "Semantic embeddings unavailable — install `.[semantic]` to build the map."
        )
        return

    options = ["(none — just explore the map)", *_course_labels()]
    choice = st.selectbox(
        "Highlight a seed course and its recommendations", options, key="map_seed"
    )
    if choice == options[0]:
        st.altair_chart(_map_base_chart(frame), use_container_width=True)
    else:
        k = st.slider("Recommendations to highlight (k)", 1, _MAX_K, 10, key="map_k")
        seed_id = _label_to_id(choice)
        recs = _fitted(DEFAULT_TECHNIQUE).recommend_similar(seed_id, k=k)
        rec_ids = {r.course_id for r in recs}
        st.altair_chart(
            _map_overlay_chart(frame, seed_id, rec_ids), use_container_width=True
        )
        st.caption(
            f"🔴 {seed_id} (seed)  ·  🔺 its top-{len(rec_ids)} SBERT recommendations  "
            "·  grey = the rest of the catalog. Nearby points share embedding-space "
            "neighborhoods, so good recommendations cluster around the seed."
        )
    st.caption(
        f"{len(frame):,} courses · projector: {used} · seed={RANDOM_SEED} "
        "(reproducible) · projection cached to `artifacts/map/`."
    )


def main() -> None:
    """Render the sidebar nav and dispatch to the selected view."""
    st.set_page_config(page_title="Course Recommender Lab", layout="wide")
    st.title("Course Recommender Lab")
    st.caption(
        "Content-based recommenders over the UC Berkeley catalog "
        "(~11k courses) — explore, compare, and rank techniques."
    )
    views = {
        "Explore": _view_explore,
        "Compare": _view_compare,
        "Leaderboard": _view_leaderboard,
        "Map": _view_map,
    }
    view = st.sidebar.radio("View", list(views))
    views[view]()


if __name__ == "__main__":
    main()
