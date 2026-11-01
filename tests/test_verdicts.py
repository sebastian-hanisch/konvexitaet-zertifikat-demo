"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np
import pytest

import kvz_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.pct(0.0051) == "0,51 %" and ev.pct(0.1234) == "12,3 %" and ev.pct(1.0) == "100,0 %"
    assert [ev.count_text(x) for x in (45, 4845, 9999)] == ["45", "4845", "9999"]
    assert ev.count_text(590_000) == "590 Tausend" and ev.count_text(77_000_000) == "77,0 Millionen"
    assert ev.count_text(3_838_380) == "3,8 Millionen" and ev.count_text(1.4e12) == "1,4 Billionen" and ev.count_text(5.4e20) == "5,4·10²⁰"
    assert ev.time_text(5.5) == "5,5 Sekunden" and ev.time_text(120) == "2,0 Minuten" and ev.time_text(7200) == "2,0 Stunden"
    assert ev.time_text(3 * 86400) == "3,0 Tagen" and ev.time_text(2 * 365.25 * 86400) == "2,0 Jahren"


def test_certificate_verdict_numbers():
    kind, text = ev.verdict_certificate(2.0, 1.99, 1.995, 200)     # Lücke 0,5 %
    assert kind == "success" and "Nach 200 Schritten" in text and "zwischen 1,990 und 2,000" in text and "Lücke von 0,50 %" in text
    assert "wahre Radius ist 1,995" in text and "Fehler also 0,25 %" in text
    kind, text = ev.verdict_certificate(2.0, 1.6, 1.9, 5)          # 20 %
    assert kind == "warning" and "Die Lücke ist noch groß" in text
    assert ev.verdict_certificate(0.0, 0.0, 0.0, 3)[0] == "success"       # eine Wolke aus einem Punkt


def test_certificate_boundaries_around_one_and_ten_percent():
    assert ev.verdict_certificate(100.0, 99.1, 99.5, 1)[0] == "success"    # 0,9 %
    assert ev.verdict_certificate(100.0, 98.9, 99.5, 1)[0] == "info"       # 1,1 %
    assert ev.verdict_certificate(100.0, 90.5, 95.0, 1)[0] == "info"       # 9,5 %
    assert ev.verdict_certificate(100.0, 89.5, 95.0, 1)[0] == "warning"    # 10,5 %


def test_trace_verdict_names_the_ratio_and_its_inverse():
    kind, text = ev.verdict_trace(0.5)
    assert kind == "info" and "**0,50-Fache der Lücke**" in text and "Faktor 2,0" in text
    assert "Faktor 4,0" in ev.verdict_trace(0.25)[1]
    assert "unendlich" in ev.verdict_trace(0.0)[1]
    assert "keinen Verlauf" in ev.verdict_trace(float("nan"))[1]


def _rate_rows():
    return [
        {"n": 50, "d": 2, "steps": {0.1: 8.0, 0.01: 80.0, 0.001: 800.0}, "product": {0.1: 0.8, 0.01: 0.8, 0.001: 0.8}},
        {"n": 1000, "d": 10, "steps": {0.1: 7.0, 0.01: 70.0, 0.001: 700.0}, "product": {0.1: 0.7, 0.01: 0.7, 0.001: 0.7}},
    ]


def test_rate_verdict_names_range_and_sizes():
    kind, text = ev.verdict_rate(_rate_rows())
    assert kind == "info"
    assert "relativen Lücke 0,10 %" in text and "700 bis 800 Schritte" in text and "0,70 bis 0,80 mal 1/ε" in text and "n·d von 100 bis 10000" in text


def test_rate_verdict_skips_cells_that_did_not_reach_the_gap():
    rows = _rate_rows()
    rows[0]["product"][0.001] = float("nan")
    assert "0,70 bis 0,70 mal 1/ε" in ev.verdict_rate(rows)[1]


def _local(hit, worst=0.5):
    return {"starts": 100, "walk_share_1pct": 1.0, "walk_max_err": 0.0005, "trim_best": 1.756, "trim_share_hit": hit, "trim_worst_over": worst, "trim_distinct": 11}


def test_local_verdict_threshold_and_numbers():
    kind, text = ev.verdict_local(_local(0.07, 0.17), 16)
    assert kind == "warning"
    for part in ("100,0 % der 100 Starts", "schlechtester Start 0,05 %", "mindestens 16 Punkte", "nur 7,0 % der Starts", "Optimum 1,756", "17,0 % darüber", "11 verschiedene Endwerte"):
        assert part in text, part
    assert ev.verdict_local(_local(0.49), 16)[0] == "warning"
    kind, text = ev.verdict_local(_local(0.5), 16)
    assert kind == "info" and "ein Beweis, dass es das Optimum ist, fehlt trotzdem" in text


def _proof_rows():
    return [
        {"n": 20, "subsets": 4845, "per_subset": 0.0023, "seconds": 11.1, "support": {2: 1330, 3: 6000, 5: 60000, 10: 785_000}},
        {"n": 40, "subsets": 3_838_380, "per_subset": 0.004, "seconds": 15353.0, "support": {2: 10660, 3: 100000, 5: 5_000_000, 10: 2.3e9}},
    ]


def test_proof_verdict_picks_the_nearest_size_and_names_the_cost():
    kind, text = ev.verdict_proof(_proof_rows(), 4, 22, 2000)
    assert kind == "warning" and "nach dem Entfernen von 4 Punkten" in text and "bei n = 20 Punkten 4845 Kugeln" in text and "2,3 ms je Kugel" in text
    assert "11,1 Sekunden" in text and "höchstens drei Punkte: 1330" in text and "in 10 Dimensionen wären es 785 Tausend" in text
    text40 = ev.verdict_proof(_proof_rows(), 6, 38, 2000)[1]
    assert "bei n = 40 Punkten 3,8 Millionen" in text40 and "4,3 Stunden" in text40


def test_playground_verdict_accepts_the_order_and_flags_a_violation():
    kind, text = ev.verdict_playground({"lower": 1.0, "exact": 2.0, "upper": 3.0})
    assert kind == "success" and "1,000 ≤ 2,000 ≤ 3,000" in text
    assert ev.verdict_playground({"lower": 2.0, "exact": 2.0, "upper": 2.0})[0] == "success"        # Gleichheit im Optimum
    assert ev.verdict_playground({"lower": 2.5, "exact": 2.0, "upper": 3.0})[0] == "error"
    assert ev.verdict_playground({"lower": 1.0, "exact": 2.0, "upper": 1.5})[0] == "error"


def test_error_gap_ratio_by_hand():
    upper, lower = np.array([4.0, 3.0, 2.5]), np.array([0.0, 2.0, 2.5])     # R* = 2,5
    ratios = ev.error_gap_ratios(upper, lower, 2.5)                          # der Zustand ohne Lücke wird übergangen
    assert ratios.tolist() == pytest.approx([1.5 / 4.0, 0.5 / 1.0])
    assert len(ev.error_gap_ratios(np.array([2.5, 2.5]), np.array([2.5, 2.5]), 2.5)) == 0
