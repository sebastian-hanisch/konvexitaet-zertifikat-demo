"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import kvz_constants as C
import kvz_evaluation as ev
import kvz_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert preset["cloud"] in C.CLOUDS
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.R_MIN <= preset["r"] <= C.R_MAX
        assert C.STEPS_MIN <= preset["steps"] <= C.STEPS_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_the_trimmed_ball_always_has_at_least_two_points():
    assert C.N_MIN - C.R_MAX >= 2


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall: Normalwolke"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_each_preset_answers_the_question_in_its_help():
    """Die Hilfetexte nennen Zahlen der Presets: sie müssen zu den Werten passen."""
    assert C.PRESETS["Wenige Schritte (5)"]["steps"] == 5
    assert C.PRESETS["Ein Haufen, vier Ausreißer"]["cloud"] == "Haufen + Ausreißer" and C.PRESETS["Ein Haufen, vier Ausreißer"]["r"] == 4 == C.PRESETS["Ein Haufen, vier Ausreißer"]["n"] - 16
    assert C.PRESETS["Zwei Haufen"]["cloud"] == "Zwei Haufen"
    big = C.PRESETS["Beweis wird unbezahlbar"]
    import kvz_algorithm as A
    assert A.count_removals(big["n"], big["r"]) == 3_838_380 > C.BRUTE_FORCE_LIMIT        # „knapp 3,8 Millionen Teilmengen“


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["cloud"] = "Zwei Haufen"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["steps"] = "abc"        # keine Zahl: bleibt Standard
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="cloud_select").value == "Zwei Haufen"
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="steps_slider").value == C.DEFAULT_STEPS
    bad = AppTest.from_file(APP, default_timeout=180)
    bad.query_params["cloud"] = "Kreis"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="cloud_select").value == C.DEFAULT_CLOUD
