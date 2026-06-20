"""Phase 8 minimal UI (`docs/roadmap/recommender_plan.md` §4).

A thin Streamlit front-end over the fitted :class:`~courserec.interfaces.Recommender`
techniques and the leaderboard. The Streamlit entrypoint is
:mod:`app.streamlit_app`; the technique registry it draws from lives in
:mod:`app.registry` and is deliberately import-safe (no Streamlit) so it can be
unit-tested on its own.
"""
