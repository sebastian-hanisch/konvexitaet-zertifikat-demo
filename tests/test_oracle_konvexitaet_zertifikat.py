"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Kernroutinen gegen einen anderen Rechenweg.

- kleinste Kugel (Welzl) gegen scipy-SLSQP auf der Epigraph-Form und gegen die Aufzählung aller Stützpunkt-Kugeln
- Dual (Frank-Wolfe) gegen scipy-SLSQP auf dem Simplex
- Eigenschaften (Invarianz, schwache Dualität, Monotonie), die auch ein Fehler in beiden Rechenwegen nicht überlebt
- getrimmt: Aufzählen der entfernten Punkte gegen Stützpunkte; lokale Suche nie besser als das Optimum

Jeder Test läuft in unter 10 s. Fehlt scipy im Test-venv, überspringen sich die scipy-Tests (pytest.importorskip), sie raten nie."""

import itertools
import math

import numpy as np
import pytest

import kvz_algorithm as A


def _cloud(seed, n, d=2):
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, d))


def _all_support_balls(pts):
    """Unabhängige exakte kleinste Kugel in der Ebene: jede Kugel durch zwei Punkte (Durchmesser) oder drei Punkte (Umkreis) bilden, die kleinste nehmen, die alles enthält."""
    best = math.inf
    n = len(pts)
    for i, j in itertools.combinations(range(n), 2):
        c, r = (pts[i] + pts[j]) / 2, np.linalg.norm(pts[i] - pts[j]) / 2
        if r < best and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = r
    for i, j, k in itertools.combinations(range(n), 3):
        a, b, c3 = pts[i], pts[j], pts[k]
        d = 2 * (a[0] * (b[1] - c3[1]) + b[0] * (c3[1] - a[1]) + c3[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            continue
        ux = ((a @ a) * (b[1] - c3[1]) + (b @ b) * (c3[1] - a[1]) + (c3 @ c3) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c3[0] - b[0]) + (b @ b) * (a[0] - c3[0]) + (c3 @ c3) * (b[0] - a[0])) / d
        c = np.array([ux, uy])
        r = np.linalg.norm(a - c)
        if r < best and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = r
    return best


def test_welzl_matches_the_circle_through_two_or_three_points():
    for seed in range(40):
        pts = _cloud(seed, 6 + seed % 10)
        assert A.smallest_ball(pts)[1] == pytest.approx(_all_support_balls(pts), rel=1e-9), seed


def test_welzl_is_independent_of_the_point_order():
    for seed in range(20):
        pts = _cloud(seed, 12)
        order = np.random.default_rng(seed + 100).permutation(12)
        a, b = A.smallest_ball(pts), A.smallest_ball(pts, order)
        assert a[1] == pytest.approx(b[1], rel=1e-9) and a[0] == pytest.approx(b[0], abs=1e-8)


def test_welzl_matches_support_enumeration_in_three_dimensions():
    """Dritte Dimension: Welzl gegen das Aufzählen aller Kugeln durch höchstens vier Punkte (anderer Suchweg, in 3D kein Kreis-Formelweg wie oben)."""
    for seed in range(10):
        pts = _cloud(seed, 9, 3)
        assert A.smallest_ball(pts)[1] == pytest.approx(A.trimmed_exact_support(pts, len(pts))[0], rel=1e-9), seed


def test_welzl_matches_scipy_in_two_and_three_dimensions():
    """SLSQP auf der Epigraph-Form in t = R^2 mit analytischer Jacobi-Matrix. Mit ftol = 1e-14 scheiterte mehr als die Hälfte („Positive directional derivative“,
    Phase 0 sah schon 0,1 bis 2,1 % Ausfälle): Standardtoleranz, und nur gemeldete Erfolge werden verglichen (mindestens 22 von 24)."""
    optimize = pytest.importorskip("scipy.optimize")
    solved = 0
    for seed in range(24):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 12, d)
        c0 = pts.mean(axis=0)
        z0 = np.concatenate([c0, [((pts - c0) ** 2).sum(axis=1).max() + 0.1]])
        cons = [{"type": "ineq", "fun": lambda z: z[-1] - ((pts - z[:-1]) ** 2).sum(axis=1), "jac": lambda z: np.hstack([2 * (pts - z[:-1]), np.ones((len(pts), 1))])}]
        res = optimize.minimize(lambda z: z[-1], z0, jac=lambda z: np.eye(len(z))[-1], method="SLSQP", constraints=cons, options={"maxiter": 1000})
        if not res.success:
            continue
        solved += 1
        assert A.smallest_ball(pts)[1] == pytest.approx(math.sqrt(abs(res.x[-1])), rel=1e-5), seed
    assert solved >= 22


def test_dual_optimum_equals_the_squared_radius_scipy():
    """Starke Dualität: max über den Simplex des Duals = R*^2, mit scipy auf dem Simplex gerechnet (anderer Weg als Frank-Wolfe); nur gemeldete Erfolge zählen."""
    optimize = pytest.importorskip("scipy.optimize")
    solved = 0
    for seed in range(12):
        pts = _cloud(seed, 8 + seed % 6)
        q = (pts * pts).sum(axis=1)
        n = len(pts)
        res = optimize.minimize(lambda lam: -(lam @ q - (lam @ pts) @ (lam @ pts)), np.full(n, 1.0 / n), method="SLSQP", bounds=[(0, 1)] * n,
                                constraints=[{"type": "eq", "fun": lambda lam: lam.sum() - 1.0}], options={"maxiter": 1000})
        if not res.success:
            continue
        solved += 1
        assert -res.fun == pytest.approx(A.smallest_ball(pts)[1] ** 2, rel=1e-5), seed
    assert solved >= 10


def test_both_bounds_bracket_the_true_radius_at_every_step():
    for seed in range(12):
        pts = _cloud(seed, 15 + seed)
        r = A.smallest_ball(pts)[1]
        run = A.dual_run(pts, 150)
        assert np.all(run.lower <= r * (1 + 1e-9)) and np.all(run.upper >= r * (1 - 1e-9)), seed
        assert A.check_certificate(pts, run.lam, run.centre) == [], seed


def test_weak_duality_on_random_pairs():
    """Schwache Dualität auf beliebigen (nicht verfahrensgenerierten) Paaren: jedes c gibt eine obere, jedes lam im Simplex eine untere Schranke."""
    rng = np.random.default_rng(0)
    for d in (2, 5):
        pts = _cloud(d, 25, d)
        r = A.smallest_ball(pts)[1] if d <= 3 else A.dual_run(pts, 4000).upper[-1]
        for _ in range(400):
            lam = rng.dirichlet(np.full(25, rng.choice([0.05, 0.3, 1.0])))
            c = pts.mean(axis=0) + rng.normal(0.0, rng.choice([0.01, 0.3, 2.0]), d)
            assert A.lower_radius(pts, lam) <= r * (1 + 1e-6)
            assert A.upper_radius(pts, c) >= r * (1 - 1e-6)


def test_invariance_under_scaling_shifting_and_permutation():
    pts = _cloud(5, 14)
    c, r = A.smallest_ball(pts)
    c2, r2 = A.smallest_ball(3.0 * pts + np.array([7.0, -2.0]))
    assert r2 == pytest.approx(3.0 * r, rel=1e-9) and c2 == pytest.approx(3.0 * c + np.array([7.0, -2.0]), abs=1e-8)
    run, run3 = A.dual_run(pts, 80), A.dual_run(3.0 * pts + 5.0, 80)
    assert run3.lower == pytest.approx(3.0 * run.lower, rel=1e-9) and run3.upper == pytest.approx(3.0 * run.upper, rel=1e-9)   # dieselben Gewichte, dieselben Schritte


def test_trimmed_enumeration_equals_support_balls():
    """Zwei exakte Wege zum selben Optimum: Aufzählen der entfernten Punkte gegen Kugeln durch höchstens d + 1 Stützpunkte."""
    rng = np.random.default_rng(1)
    for seed in range(30):
        n = int(rng.integers(6, 11))
        r = int(rng.integers(1, 4))
        pts = _cloud(seed, n)
        assert A.trimmed_exact_support(pts, n - r)[0] == pytest.approx(A.trimmed_exact(pts, r, 1000)[0], rel=1e-9, abs=1e-12), (seed, n, r)


def test_trimmed_enumeration_equals_support_balls_in_three_dimensions():
    for seed in range(6):
        pts = _cloud(seed, 8, 3)
        assert A.trimmed_exact_support(pts, 6)[0] == pytest.approx(A.trimmed_exact(pts, 2, 100)[0], rel=1e-9)


def test_local_search_is_never_better_than_the_optimum_and_never_worse_than_its_start():
    rng = np.random.default_rng(2)
    for seed in range(20):
        pts = _cloud(seed, 14)
        m = 10
        best, _ = A.trimmed_exact_support(pts, m)
        for _ in range(5):
            c0 = rng.uniform(-3, 3, 2)
            start = A.smallest_ball(pts[np.argsort(((pts - c0) ** 2).sum(axis=1), kind="stable")[:m]])[1]
            _, r = A.trimmed_local(pts, m, c0)
            assert best * (1 - 1e-9) <= r <= start * (1 + 1e-9), seed


def test_trimmed_ball_really_contains_m_points():
    for seed in range(10):
        pts = _cloud(seed, 12)
        r, c = A.trimmed_exact_support(pts, 9)
        assert (np.linalg.norm(pts - c, axis=1) <= r * (1 + 1e-9) + 1e-12).sum() >= 9
        assert r <= A.smallest_ball(pts)[1] + 1e-12


def test_steps_to_gap_agrees_with_an_independent_replay():
    """Die Zählfunktion gegen ein eigenes Nachspielen mit `dual_run`: erste Stelle, an der die Lücke klein genug ist."""
    for seed in range(8):
        pts = _cloud(seed, 30, 5)
        run = A.dual_run(pts, 800)
        gap = (run.upper - run.lower) / run.upper
        for eps in (0.1, 0.02):
            hits = np.nonzero(gap <= eps)[0]
            assert A.steps_to_gap(pts, eps, 800) == (int(hits[0]) if len(hits) else None), (seed, eps)
