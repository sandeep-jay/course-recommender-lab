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

import pandas as pd
import streamlit as st

from app.registry import (
    DEFAULT_TECHNIQUE,
    course_label,
    make_recommender,
    technique_names,
)
from courserec.config import PLOTS_DIR, RESULTS_DIR
from courserec.data import load_processed
from courserec.interfaces import Rec, Recommender
from courserec.recommenders.llm import RecommendationExplainer

LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"
EMBEDDING_MAP_PNG = PLOTS_DIR / "embedding_map.png"
_MAX_K = 20


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
    tech_b = col_b.selectbox(
        "Technique B", names, index=min(1, len(names) - 1), key="compare_b"
    )
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
        st.caption(
            f"{len(board)} technique×config rows from `{LEADERBOARD_CSV.name}` — "
            "click a column header to sort. Regenerate with "
            "`python scripts/run_eval.py`."
        )
        st.dataframe(board, use_container_width=True, hide_index=True)
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


def main() -> None:
    """Render the sidebar nav and dispatch to the selected view."""
    st.set_page_config(page_title="Course Recommender Lab", layout="wide")
    st.title("Course Recommender Lab")
    st.caption(
        "Content-based recommenders over the UC Berkeley catalog "
        "(~11k courses) — explore, compare, and rank techniques."
    )
    view = st.sidebar.radio("View", ["Explore", "Compare", "Leaderboard"])
    if view == "Explore":
        _view_explore()
    elif view == "Compare":
        _view_compare()
    else:
        _view_leaderboard()


if __name__ == "__main__":
    main()
