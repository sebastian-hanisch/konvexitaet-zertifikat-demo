"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über die festen Wolken SWEEP_SEEDS, nie über einen einzelnen Seed als Befund; die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo ein Zufallsstrom oder eine Bibliotheksversion die letzte Stelle bewegt (CI installiert ungepinnt, Linux rundet anders als Windows).
"""

import math
from statistics import median

import numpy as np
import pytest

import kvz_algorithm as A
import kvz_constants as C
import kvz_evaluation as ev
from kvz_scenario import make_gaussian


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def rate_rows():
    return ev.rate_experiment()


@pytest.fixture(scope="module")
def local_runs():
    """Je Wolkenart die Ergebnisse über die fünf festen Wolken (100 Starts je Wolke, wie im README)."""
    return {kind: [ev.local_experiment(ev.Settings(cloud=kind, seed=seed)) for seed in C.SWEEP_SEEDS] for kind in C.CLOUDS}


# ----------------------------------------------------------------------------- Handinstanz und Standardwolke

def test_hand_instance_numbers():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    c, r = A.smallest_ball(pts)
    assert c == pytest.approx([2.0, 1.5]) and r == pytest.approx(2.5)
    assert A.dual_value(pts, np.array([0.0, 0.5, 0.5, 0.0])) == pytest.approx(6.25)
    trap = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
    assert A.trimmed_exact_support(trap, 3)[0] == pytest.approx(math.sqrt(2) / 2, abs=1e-12)
    assert A.trimmed_local(trap, 3, np.array([10.0, 0.0]))[1] == pytest.approx(5.0)


def test_standard_cloud_numbers(std):
    assert std.upper == pytest.approx(2.346, abs=0.002) and std.lower == pytest.approx(2.334, abs=0.002) and std.exact_r == pytest.approx(2.340, abs=0.002)
    assert std.gap_rel == pytest.approx(0.0051, abs=0.0005)
    assert (std.upper - std.exact_r) / std.exact_r == pytest.approx(0.0027, abs=0.0005)
    assert std.problems == []


# ----------------------------------------------------------------------------- Zertifikat

def test_error_is_about_half_the_gap_for_every_cloud_kind():
    """README: wahrer Fehler der oberen Schranke ≈ ½ Lücke: Median über den Verlauf in 14 von 15 festen Wolken 0,500 bis 0,503, in einer (Normalwolke, Seed 100001) 0,59."""
    medians = []
    for kind in C.CLOUDS:
        ms = [ev.ratio_median(ev.analyse(ev.Settings(cloud=kind, seed=seed))) for seed in C.SWEEP_SEEDS]
        assert median(ms) == pytest.approx(0.5, abs=0.01), kind
        assert all(0.45 <= x <= 0.65 for x in ms), (kind, ms)
        medians += ms
    assert sum(abs(x - 0.5) <= 0.01 for x in medians) >= 14 and max(medians) == pytest.approx(0.59, abs=0.02)


def test_error_is_about_half_the_gap_in_ten_dimensions():
    """Zehn Dimensionen, n = 100 (README: 0,48 bis 0,50): Referenz R* aus einem langen Lauf (Mitte beider Schranken)."""
    for seed in C.SWEEP_SEEDS:
        pts = make_gaussian(100, 10, seed)
        fine = A.dual_run(pts, 20000)
        r_ref = (fine.upper[-1] + fine.lower[-1]) / 2
        run = A.dual_run(pts, 2000)
        assert 0.4 <= float(np.median(ev.error_gap_ratios(run.upper, run.lower, r_ref))) <= 0.6, seed


def test_weak_duality_has_no_violation_on_random_pairs():
    rng = np.random.default_rng(11)
    bad = 0
    for d in (2, 10, 50):
        pts = make_gaussian(60, d, d)
        r = A.smallest_ball(pts)[1] if d <= 3 else A.dual_run(pts, 6000).upper[-1]
        slack = 1e-3 if d > 3 else 1e-9          # ab d > 3 ist r nur die obere Schranke eines langen Laufs
        for _ in range(500):
            lam = rng.dirichlet(np.ones(60) * rng.choice([0.05, 0.3, 1.0]))
            c = pts.mean(axis=0) + rng.normal(0.0, rng.choice([0.01, 0.3, 2.0]), d)
            bad += A.lower_radius(pts, lam) > r * (1 + slack) or A.upper_radius(pts, c) < r * (1 - slack)
    assert bad == 0


# ----------------------------------------------------------------------------- Rate

def test_steps_scale_like_one_over_eps_independent_of_size(rate_rows):
    """README: etwa 0,5 bis 0,9 mal 1/ε Schritte, für n·d von 100 bis 20 000."""
    assert [(r["n"], r["d"]) for r in rate_rows] == list(C.RATE_SIZES)
    for row in rate_rows:
        for eps in C.RATE_EPS:
            assert 0.45 <= row["product"][eps] <= 0.95, (row["n"], row["d"], eps, row["steps"][eps])
    finest = [r["steps"][0.001] for r in rate_rows]
    assert 600 <= min(finest) and max(finest) <= 950 and max(finest) / min(finest) < 1.5          # 682 bis 866 gemessen
    sizes = sorted(rate_rows, key=lambda r: r["n"] * r["d"])
    assert sizes[0]["n"] * sizes[0]["d"] == 100 and sizes[-1]["n"] * sizes[-1]["d"] == 20000


def test_ten_times_more_accuracy_costs_about_ten_times_more_steps(rate_rows):
    """README: Verhältnis 9 bis 12 von ε = 0,01 auf 0,001 und 7 bis 11 von 0,1 auf 0,01."""
    for row in rate_rows:
        assert 8.5 <= row["steps"][0.001] / row["steps"][0.01] <= 12.5, (row["n"], row["d"])
        assert 6.5 <= row["steps"][0.01] / row["steps"][0.1] <= 11.5, (row["n"], row["d"])


# ----------------------------------------------------------------------------- lokal gegen global

def test_convex_walk_ends_at_the_true_radius_from_every_start(local_runs):
    """README: das Mittelpunkt-Verfahren endet nach 2000 Schritten von jedem Start höchstens 0,1 % über dem wahren Radius."""
    for kind, runs in local_runs.items():
        for res in runs:
            assert res["walk_share_1pct"] == 1.0 and res["walk_max_err"] < 0.001, kind


def test_trimmed_local_search_misses_the_optimum_most_of_the_time(local_runs):
    """README: Median der Trefferquote je Wolkenart 10 / 29 / 25 % (Normalwolke / Haufen + Ausreißer / Zwei Haufen), über alle 15 Wolken höchstens 40 %."""
    medians = {kind: median(r["trim_share_hit"] for r in runs) for kind, runs in local_runs.items()}
    assert medians["Normalwolke"] == pytest.approx(0.10, abs=0.08)
    assert medians["Haufen + Ausreißer"] == pytest.approx(0.29, abs=0.10)
    assert medians["Zwei Haufen"] == pytest.approx(0.25, abs=0.10)
    assert max(r["trim_share_hit"] for runs in local_runs.values() for r in runs) <= 0.55
    assert min(r["trim_share_hit"] for runs in local_runs.values() for r in runs) <= 0.05             # es gibt Wolken, in denen fast kein Start trifft
    assert max(r["trim_share_hit"] for runs in local_runs.values() for r in runs) >= 0.3              # und solche, in denen ein knappes Drittel trifft („0 bis 40 %“)


def test_trimmed_local_search_worst_start_and_number_of_end_values(local_runs):
    worst = {kind: median(r["trim_worst_over"] for r in runs) for kind, runs in local_runs.items()}
    assert 0.2 <= worst["Normalwolke"] <= 0.5 and 1.5 <= worst["Haufen + Ausreißer"] <= 3.5 and 0.05 <= worst["Zwei Haufen"] <= 0.2
    assert max(r["trim_worst_over"] for runs in local_runs.values() for r in runs) == pytest.approx(2.77, abs=0.5)        # „im Extremfall 277 %“
    medians = [median(r["trim_distinct"] for r in runs) for runs in local_runs.values()]
    assert 6 <= min(medians) and max(medians) <= 10                                                                    # „6 bis 9 verschiedene Endwerte“


# ----------------------------------------------------------------------------- Aufwand des Beweises

def test_proof_counts():
    assert A.count_removals(20, 4) == 4845 and A.count_removals(40, 6) == 3_838_380
    assert A.count_support_candidates(20, 2) == 1330 and A.count_support_candidates(40, 2) == 10_660
    assert A.count_support_candidates(20, 10) == 784_605
    assert A.count_support_candidates(40, 10) > 2e9


def test_support_balls_need_fewer_balls_than_the_enumeration_and_agree(std):
    """README: in der Ebene prüft der Weg über Stützpunkte 1330 Kugeln statt 4845 Teilmengen (n = 20, r = 4) und findet dasselbe Optimum (hier auf n = 12, r = 3 nachgerechnet)."""
    assert A.count_support_candidates(20, 2) < A.count_removals(20, 4)
    pts = std.inst.points[:12]
    r_support, _ = A.trimmed_exact_support(pts, 9)
    r_enum, _, count = A.trimmed_exact(pts, 3, 300)
    assert count == 220 and r_support == pytest.approx(r_enum, rel=1e-9)
