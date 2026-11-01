"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kvz_algorithm as A
from kvz_scenario import make_cloud


def test_single_point():
    c, r = A.smallest_ball(np.array([[3.0, -1.0]]))
    assert c.tolist() == [3.0, -1.0] and r == 0.0
    run = A.dual_run(np.array([[3.0, -1.0]]), 5)
    assert run.upper.tolist() == [0.0] * 6 and run.lower.tolist() == [0.0] * 6


def test_exactly_two_points_give_the_midpoint():
    """n = 2: der kleinste Fall mit etwas zu umschließen; die Kugel liegt auf der Mitte und hat den halben Abstand."""
    pts = np.array([[0.0, 0.0], [6.0, 8.0]])
    c, r = A.smallest_ball(pts)
    assert c == pytest.approx([3.0, 4.0]) and r == pytest.approx(5.0)
    run = A.dual_run(pts, 5)
    assert run.upper[-1] == pytest.approx(5.0) and run.lower[-1] == pytest.approx(5.0)


def test_identical_points_have_radius_zero_and_do_not_divide_by_zero():
    pts = np.array([[1.0, 1.0]] * 5)
    assert A.smallest_ball(pts)[1] == 0.0
    run = A.dual_run(pts, 10)
    assert np.all(run.upper == 0.0) and np.all(run.lower == 0.0)
    assert A.trimmed_local(pts, 3, np.array([5.0, 5.0]))[1] == 0.0


def test_three_collinear_points_use_only_the_outer_two():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    c, r = A.smallest_ball(pts)
    assert c == pytest.approx([1.0, 1.0]) and r == pytest.approx(2 ** 0.5)


def test_zero_steps_return_the_start_state():
    pts = np.array([[0.0, 0.0], [4.0, 0.0]])
    run = A.dual_run(pts, 0)
    assert len(run.upper) == 1 and run.upper[0] == pytest.approx(4.0) and run.lower[0] == 0.0 and run.lam.tolist() == [1.0, 0.0]


def test_invalid_inputs_raise_with_a_message():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="leere Punktmenge"):
        A.smallest_ball(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="mindestens ein Punkt"):
        A.dual_run(np.zeros((0, 2)), 3)
    with pytest.raises(ValueError, match="Schrittzahl"):
        A.dual_run(pts, -1)
    for m in (0, 4):
        with pytest.raises(ValueError, match="außerhalb von 1 bis 3"):
            A.trimmed_local(pts, m, np.zeros(2))
        with pytest.raises(ValueError, match="außerhalb von 1 bis 3"):
            A.trimmed_exact_support(pts, m)
    with pytest.raises(ValueError, match="außerhalb von 0 bis 2"):
        A.trimmed_exact(pts, 3, 100)
    with pytest.raises(ValueError, match="außerhalb von 0 bis 2"):
        A.trimmed_exact(pts, -1, 100)
    with pytest.raises(ValueError, match="erlaubt sind 2"):
        A.trimmed_exact(pts, 1, 2)


def test_bounds_of_the_trimmed_ball_are_inclusive(mini):
    """m = n: dieselbe Kugel wie ohne Aussortieren; m = 1: Radius null (ein Punkt genügt); r = 0 in der Aufzählung gleich m = n."""
    c, r = A.smallest_ball(mini)
    assert A.trimmed_exact_support(mini, len(mini))[0] == pytest.approx(r)
    assert A.trimmed_exact_support(mini, 1)[0] == 0.0
    assert A.trimmed_exact(mini, 0, 10)[0] == pytest.approx(r)
    assert A.trimmed_exact(mini, 3, 10)[0] == 0.0
    assert A.trimmed_local(mini, len(mini), np.array([9.0, 9.0]))[1] == pytest.approx(r)


def test_clouds_are_deterministic_and_validated():
    a, b = make_cloud("Zwei Haufen", 12, 3, 7), make_cloud("Zwei Haufen", 12, 3, 7)
    assert np.array_equal(a.points, b.points) and a.n == 12
    assert not np.array_equal(a.points, make_cloud("Zwei Haufen", 12, 3, 8).points)
    with pytest.raises(ValueError, match="unbekannte Wolke"):
        make_cloud("Kreis", 10, 2, 1)


def test_outlier_cloud_puts_exactly_r_points_far_right():
    pts = make_cloud("Haufen + Ausreißer", 20, 4, 35).points
    assert pts[:16].max() < 5.0 and (pts[16:, 0] > 0.9).all()          # die ersten n − r Punkte sind der Haufen, die letzten r liegen rechts davon
    assert make_cloud("Haufen + Ausreißer", 8, 6, 3).n == 8
