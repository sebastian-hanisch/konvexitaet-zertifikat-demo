"""Abbildungen: Achsen gesperrt, Legende zum Inhalt passend, Verlaufsdiagramme ohne Fehler an den Rändern."""

import numpy as np
import pytest

import kvz_algorithm as A
import kvz_constants as C
import kvz_evaluation as ev
import kvz_visualization as V


def _names(fig):
    return [t.name for t in fig.data]


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(steps=50))


def test_cloud_legend_mentions_the_dual_weight_only_when_there_are_weights(a):
    with_weights = V.build_cloud(a.inst.points, a.run.centre, a.upper, a.run.lam, a.exact_centre, a.exact_r)
    without = V.build_cloud(a.inst.points, trimmed=(a.exact_centre, 1.0), exact_centre=a.exact_centre, exact_r=a.exact_r)
    assert "Punkte (Größe = Gewicht im Dual)" in _names(with_weights) and "Punkte" in _names(without)
    assert "obere Schranke (Kugel um c)" in _names(with_weights) and "getrimmte Kugel" in _names(without)
    assert _locked(with_weights) and _locked(without) and with_weights.layout.yaxis.scaleanchor == "x"


def test_marker_size_grows_with_the_weight(a):
    fig = V.build_cloud(a.inst.points, a.run.centre, a.upper, a.run.lam)
    sizes = np.asarray(next(t for t in fig.data if t.name.startswith("Punkte")).marker.size)
    assert sizes.min() == pytest.approx(7.0) and sizes.max() == pytest.approx(7.0 + 40.0 * a.run.lam.max())


def test_trace_charts_are_locked_and_log_scaled(a):
    for fig in (V.build_bounds(a.run, a.exact_r), V.build_gap(a.run, a.exact_r)):
        assert _locked(fig) and fig.layout.xaxis.type == "log"
    assert V.build_gap(a.run, a.exact_r).layout.yaxis.type == "log"


def test_gap_chart_stays_finite_when_the_gap_is_zero():
    pts = np.array([[0.0, 0.0], [4.0, 0.0]])
    run = A.dual_run(pts, 6)                      # nach einem Schritt ist die Lücke null
    fig = V.build_gap(run, 2.0)
    assert all(np.all(np.isfinite(t.y)) and np.all(np.asarray(t.y) > 0) for t in fig.data)


def test_rate_local_and_proof_charts_build(a):
    rate = V.build_rate(ev.rate_experiment(sizes=((20, 2),), seeds=(1, 2), max_steps=400))
    assert _locked(rate) and rate.layout.xaxis.autorange == "reversed" and _names(rate)[-1] == "Referenz 0,75/ε"
    res = ev.local_experiment(ev.Settings(), starts=5, walk_steps=50)
    assert _locked(V.build_local(res))
    proof = V.build_proof(ev.proof_experiment(ev.Settings(), sizes=(10, 12), dims=C.PROOF_DIMS), 4)
    assert _locked(proof) and proof.layout.yaxis.type == "log" and len(proof.data) == 1 + len(C.PROOF_DIMS)
