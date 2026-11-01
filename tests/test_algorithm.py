"""Kern gegen Handrechnung auf der Mini-Instanz: je Einheit eine Regel (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import kvz_algorithm as A


def test_upper_radius_is_the_distance_to_the_farthest_point(mini):
    assert A.upper_radius(mini, np.array([2.0, 1.5])) == pytest.approx(2.5)
    assert A.upper_radius(mini, np.array([0.0, 0.0])) == pytest.approx(4.0)


def test_dual_value_by_hand(mini):
    assert A.dual_value(mini, np.array([0.0, 0.5, 0.5, 0.0])) == pytest.approx(6.25)
    assert A.dual_value(mini, np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert A.dual_value(mini, np.array([0.5, 0.5, 0.0, 0.0])) == pytest.approx(4.0)     # zwei Punkte im Abstand 4: (4/2)^2
    assert A.lower_radius(mini, np.array([0.0, 0.5, 0.5, 0.0])) == pytest.approx(2.5)


def test_lower_radius_clips_rounding_below_zero(mini):
    assert A.lower_radius(mini, np.array([0.0, 0.0, 0.0, 1.0])) == 0.0   # ein Punkt: kein Spread, höchstens Rundungsrest unter null


def test_smallest_ball_by_hand(mini):
    c, r = A.smallest_ball(mini)
    assert c == pytest.approx([2.0, 1.5]) and r == pytest.approx(2.5)


def test_first_step_by_hand(mini):
    """Start bei (0,0): fernster Punkt (4,0), Linienminimierung gibt gam = 16 / 32 = 0,5, also Gewichte (½; ½), c = (2; 0), q_mean = 8."""
    sq = (mini * mini).sum(axis=1)
    lam, c, q, moved = A.fw_step(mini, sq, np.array([1.0, 0, 0, 0]), mini[0].copy(), float(sq[0]))
    assert moved and lam == pytest.approx([0.5, 0.5, 0, 0]) and c == pytest.approx([2.0, 0.0]) and q == pytest.approx(8.0)


def test_second_step_by_hand(mini):
    """Von c = (2; 0): fernster Punkt (0,3) (Abstand² 13), gam = 9/26."""
    sq = (mini * mini).sum(axis=1)
    lam, c, q, moved = A.fw_step(mini, sq, np.array([0.5, 0.5, 0, 0]), np.array([2.0, 0.0]), 8.0)
    gam = 9.0 / 26.0
    assert moved and lam == pytest.approx([0.5 * (1 - gam), 0.5 * (1 - gam), gam, 0]) and c == pytest.approx([2.0 - 2.0 * gam, 3.0 * gam])


def test_step_without_a_farther_point_does_not_move():
    pts = np.array([[1.0, 1.0], [1.0, 1.0]])
    lam, c, q, moved = A.fw_step(pts, (pts * pts).sum(axis=1), np.array([1.0, 0.0]), pts[0].copy(), 2.0)
    assert not moved and lam.tolist() == [1.0, 0.0]


def test_dual_run_brackets_the_radius_and_closes_the_gap(mini):
    run = A.dual_run(mini, 1000)
    assert len(run.upper) == len(run.lower) == 1001
    assert np.all(run.lower <= 2.5 + 1e-12) and np.all(run.upper >= 2.5 - 1e-12)
    assert (run.upper[-1] - run.lower[-1]) / run.upper[-1] < 0.01
    assert A.check_certificate(mini, run.lam, run.centre) == []


def test_lower_bound_never_decreases(mini):
    """Die Linienminimierung steigt auf dem konkaven Dual nie ab."""
    run = A.dual_run(mini, 300)
    assert np.all(np.diff(run.lower) >= -1e-12)


def test_check_certificate_names_each_broken_rule(mini):
    good = np.array([0.0, 0.5, 0.5, 0.0])
    centre = good @ mini
    assert A.check_certificate(mini, good, centre) == []
    assert "Gewichte negativ" in A.check_certificate(mini, np.array([-0.5, 1.0, 0.5, 0.0]), np.array([-0.5, 1.0, 0.5, 0.0]) @ mini)
    assert "Gewichte summieren sich nicht zu 1" in A.check_certificate(mini, good * 2, good * 2 @ mini)
    assert "Mittelpunkt ist nicht der Gewichtsschwerpunkt" in A.check_certificate(mini, good, centre + 1.0)


def test_check_certificate_catches_a_lower_bound_above_the_upper():
    """Gewichte (2; 2) summieren sich zu 4: der Dualwert 4·100 = 400 gibt Radius 20 über der oberen Schranke 10 (Kugel um den Ursprung). Mit Gewichten im Simplex ist das unmöglich."""
    pts = np.array([[10.0, 0.0], [-10.0, 0.0]])
    problems = A.check_certificate(pts, np.array([2.0, 2.0]), np.array([0.0, 0.0]))
    assert "untere Schranke liegt über der oberen" in problems and "Gewichte summieren sich nicht zu 1" in problems
    assert A.check_certificate(pts, np.array([0.5, 0.5]), np.array([0.0, 0.0])) == []


def test_trap_by_hand(trap):
    c, r = A.trimmed_local(trap, 3, np.array([10.0, 0.0]))
    assert r == pytest.approx(5.0) and c == pytest.approx([5.0, 0.0])
    c2, r2 = A.trimmed_local(trap, 3, np.array([0.5, 0.5]))
    assert r2 == pytest.approx(math.sqrt(2) / 2) and c2 == pytest.approx([0.5, 0.5])


def test_exact_trimmed_by_hand(trap):
    r, c = A.trimmed_exact_support(trap, 3)
    assert r == pytest.approx(math.sqrt(2) / 2) and c == pytest.approx([0.5, 0.5])
    r2, c2, count = A.trimmed_exact(trap, 1, 100)
    assert r2 == pytest.approx(math.sqrt(2) / 2) and count == 4 and c2 == pytest.approx([0.5, 0.5])


def test_counts_by_hand():
    assert A.count_removals(20, 4) == 4845 and A.count_removals(40, 6) == 3838380
    assert A.count_support_candidates(20, 2) == 190 + 1140           # Paare und Dreiecke
    assert A.count_support_candidates(4, 3) == 6 + 4 + 1             # Paare, Dreiecke, Vierlinge


def test_steps_to_gap_matches_the_run(mini):
    run = A.dual_run(mini, 400)
    gap = (run.upper - run.lower) / run.upper
    for eps in (0.2, 0.05, 0.01):
        first = int(np.argmax(gap <= eps))
        assert A.steps_to_gap(mini, eps, 400) == first
    assert A.steps_to_gap(mini, 1e-12, 5) is None


def test_centre_walk_converges_from_far_away(mini):
    for c0 in (np.array([50.0, -30.0]), np.array([-5.0, 5.0])):
        _, r = A.centre_walk(mini, c0, 3000)
        assert r == pytest.approx(2.5, rel=0.02)
