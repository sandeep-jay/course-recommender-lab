"""Tests for the Phase 8 UI technique registry (`app/registry.py`).

The registry is the import-safe core of the Streamlit app: it must list at least
one technique, default to a real one, build only genuine
:class:`~courserec.interfaces.Recommender` instances, reject unknown names, and
format course labels the way the picker round-trips them. Streamlit is never
imported here, so these run in the base test environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import registry
from courserec.interfaces import Recommender


def test_default_is_a_registered_technique() -> None:
    """The advertised default must be selectable and listed first."""
    names = registry.technique_names()
    assert names, "the registry must expose at least one technique"
    assert registry.DEFAULT_TECHNIQUE in names
    assert names[0] == registry.DEFAULT_TECHNIQUE


@pytest.mark.parametrize("name", registry.technique_names())
def test_factories_build_unfitted_recommenders(name: str) -> None:
    """Every factory yields a fresh Recommender with a non-empty name + config."""
    rec = registry.make_recommender(name)
    assert isinstance(rec, Recommender)
    assert isinstance(rec.name, str) and rec.name
    assert isinstance(rec.config, dict)


def test_make_recommender_unknown_raises() -> None:
    """An unregistered name is a KeyError, not a silent default."""
    with pytest.raises(KeyError):
        registry.make_recommender("does-not-exist")


def test_course_label_formats_id_and_title() -> None:
    """A real title produces the ``"<id> — <title>"`` label the picker splits on."""
    label = registry.course_label("AEROENG 1", "Introduction to Flight")
    assert label == "AEROENG 1 — Introduction to Flight"
    # The label round-trips back to the id on the picker's separator.
    assert label.split(" — ", 1)[0] == "AEROENG 1"


@pytest.mark.parametrize("title", [None, "", "   ", np.nan, "nan"])
def test_course_label_falls_back_to_id_on_missing_title(title: object) -> None:
    """A missing/NA title degrades to the bare id — never ``"<id> — nan"``."""
    assert registry.course_label("STAT 20", title) == "STAT 20"
