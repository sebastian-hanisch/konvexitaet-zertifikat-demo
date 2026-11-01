"""Pfade der App, die der Rauchtest nicht erreicht: Experiment 3, Live-Aufzählung, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kvz_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        (at.sidebar.selectbox if key == "cloud_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.slider)(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_experiment_3_runs_and_names_both_sides():
    at = _app(n_slider=12, r_slider=3)
    at.button(key="local_run").click().run()
    assert not at.exception
    text = _texts(at)
    assert "Konvex:" in text and "Getrimmt (mindestens 9 Punkte)" in text


def test_live_enumeration_agrees_with_the_support_balls():
    at = _app(n_slider=12, r_slider=3)            # C(12, 3) = 220 Teilmengen, unter dem Limit
    at.button(key="brute_run").click().run()
    assert not at.exception
    assert any("Aufzählung über 220 Teilmengen" in s.value and "gleich dem Ergebnis über Stützpunkte" in s.value for s in at.success)


def test_enumeration_button_is_absent_above_the_limit_and_says_why():
    at = _app()                                    # C(20, 4) = 4845 > 2000
    assert not [b for b in at.button if b.key == "brute_run"]
    assert any("4845 Teilmengen sind mehr als die 2000" in i.value for i in at.info)


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "r_slider": pick(C.R_MIN, C.R_MAX), "steps_slider": pick(C.STEPS_MIN, C.STEPS_MAX)}
    at = _app(**state)
    assert not at.exception
    assert not at.error


def test_smallest_cloud_with_the_most_removals_still_has_two_points():
    at = _app(n_slider=C.N_MIN, r_slider=C.R_MAX, steps_slider=1)
    at.button(key="local_run").click().run()
    assert not at.exception and not at.error


def test_every_cloud_kind_renders_without_error():
    for kind in C.CLOUDS:
        at = _app(cloud_select=kind)
        assert not at.error, kind
