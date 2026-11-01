"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kvz_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _circle(fig, centre, radius, color, name, dash="solid", width=2):
    """Kreis als Linienzug (nicht als Shape), damit er in der Legende steht und die Achsen sich anpassen."""
    t = np.linspace(0.0, 2.0 * math.pi, 121)
    fig.add_trace(go.Scatter(x=centre[0] + radius * np.cos(t), y=centre[1] + radius * np.sin(t), mode="lines", name=name,
                             line=dict(color=color, width=width, dash=dash), hoverinfo="skip"))


def build_cloud(points, centre=None, upper=None, lam=None, exact_centre=None, exact_r=None, trimmed=None):
    """Punkte (Größe nach Gewicht lam), Kugel um `centre` (obere Schranke, rot), exakte Kugel (blau gestrichelt), optional die getrimmte Kugel (gepunktet)."""
    fig = go.Figure()
    if exact_centre is not None:
        _circle(fig, exact_centre, exact_r, C.COLOR_EXACT, "wahrer kleinster Radius", dash="dash")
    if centre is not None:
        _circle(fig, centre, upper, C.COLOR_UPPER, "obere Schranke (Kugel um c)")
    if trimmed is not None:
        _circle(fig, trimmed[0], trimmed[1], C.COLOR_LOWER, "getrimmte Kugel", dash="dot")
    sizes = 7.0 + (0.0 if lam is None else 40.0 * np.asarray(lam))
    fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="Punkte" if lam is None else "Punkte (Größe = Gewicht im Dual)",
                             marker=dict(size=sizes, color=C.COLOR_POINT, line=dict(width=1, color="#444")),
                             hovertemplate="(%{x:.2f}; %{y:.2f})<extra></extra>"))
    if centre is not None:
        fig.add_trace(go.Scatter(x=[centre[0]], y=[centre[1]], mode="markers", name="Mittelpunkt c", marker=dict(symbol="x", size=11, color=C.COLOR_UPPER)))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1)
    return _base(fig, 440)


def build_bounds(run, exact_r):
    """Beide Schranken über die Schritte (logarithmische Schrittachse), dazu der wahre Radius als Linie."""
    t = np.arange(len(run.upper))[1:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=run.upper[1:], mode="lines", name="obere Schranke", line=dict(color=C.COLOR_UPPER, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=run.lower[1:], mode="lines", name="untere Schranke (dual)", line=dict(color=C.COLOR_LOWER, width=2.5)))
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[exact_r, exact_r], mode="lines", name="wahrer Radius", line=dict(color=C.COLOR_EXACT, width=1.5, dash="dash")))
    fig.update_xaxes(title_text="Schritt (logarithmisch)", type="log")
    fig.update_yaxes(title_text="Radius")
    return _base(fig)


def build_gap(run, exact_r):
    """Relative Lücke und relativer wahrer Fehler der oberen Schranke, beide logarithmisch."""
    t = np.arange(len(run.upper))[1:]
    gap = np.maximum((run.upper[1:] - run.lower[1:]) / np.maximum(run.upper[1:], 1e-300), 1e-12)
    err = np.maximum(run.upper[1:] / exact_r - 1.0, 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=gap, mode="lines", name="Lücke (bewiesen)", line=dict(color=C.COLOR_LOWER, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=err, mode="lines", name="wahrer Fehler (nur mit bekannter Lösung)", line=dict(color=C.COLOR_EXACT, width=1.5)))
    fig.update_xaxes(title_text="Schritt (logarithmisch)", type="log")
    fig.update_yaxes(title_text="relativ zum Radius (logarithmisch)", type="log")
    return _base(fig)


def build_rate(rows, eps_list=C.RATE_EPS):
    """Schritte bis zur relativen Lücke eps für jede Wolke (n, d); gestrichelt die Referenz 0,75/eps."""
    fig = go.Figure()
    for row in rows:
        fig.add_trace(go.Scatter(x=list(eps_list), y=[row["steps"][e] for e in eps_list], mode="lines+markers", name=f"n = {row['n']}, d = {row['d']}",
                                 hovertemplate="Lücke %{x}: %{y:.0f} Schritte<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(eps_list), y=[0.75 / e for e in eps_list], mode="lines", name="Referenz 0,75/ε", line=dict(color="#444", dash="dash", width=1.5)))
    fig.update_xaxes(title_text="relative Lücke ε (logarithmisch)", type="log", autorange="reversed")
    fig.update_yaxes(title_text="Schritte bis Lücke ≤ ε (logarithmisch)", type="log")
    return _base(fig)


def build_local(res):
    """Endradius geteilt durch den besten Radius: konvexes Mittelpunkt-Verfahren (liegt bei 1) gegen das Wechselverfahren der getrimmten Kugel."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=1.0 + res["walk_err"], name="konvex: Mittelpunkt-Verfahren", marker_color=C.COLOR_EXACT, xbins=dict(size=0.02), opacity=0.85))
    fig.add_trace(go.Histogram(x=res["trim_values"] / res["trim_best"], name="getrimmt: Wechselverfahren", marker_color=C.COLOR_UPPER, xbins=dict(size=0.02), opacity=0.85))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Endradius geteilt durch den besten Radius")
    fig.update_yaxes(title_text="Starts")
    return _base(fig)


def build_proof(rows, r, dims=C.PROOF_DIMS):
    """Kugeln, die ein Beweis durch Aufzählen prüfen müsste, über die Punktzahl (logarithmisch)."""
    ns = [row["n"] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[row["subsets"] for row in rows], mode="lines+markers", name=f"Teilmengen C(n, {r})", line=dict(color=C.COLOR_UPPER, width=2.5)))
    for d in dims:
        fig.add_trace(go.Scatter(x=ns, y=[row["support"][d] for row in rows], mode="lines+markers", name=f"Stützpunkt-Kugeln, d = {d}", line=dict(dash="dash")))
    fig.update_xaxes(title_text="Punkte n")
    fig.update_yaxes(title_text="zu prüfende Kugeln (logarithmisch)", type="log")
    return _base(fig)
