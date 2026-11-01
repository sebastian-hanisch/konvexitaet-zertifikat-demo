"""Kern der Demo (nur numpy/Standardbibliothek): kleinste umschließende Kugel mit Zertifikat und ihre nichtkonvexe, getrimmte Variante.

Primal:  min R  mit  ||x_i - c|| <= R für alle i.     Dual:  max über Gewichte lam im Simplex  sum lam_i ||x_i||^2 - ||sum lam_i x_i||^2  (= R*^2).
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); `dual_run` ruft nur der Reihe nach auf."""

import itertools
import math
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- Schranken (die beiden Seiten des Zertifikats)

def upper_radius(points: np.ndarray, c: np.ndarray) -> float:
    """Obere Schranke: Radius der Kugel um c, die alle Punkte enthält (Abstand zum fernsten Punkt). Gilt für jedes c."""
    return float(np.sqrt(((points - c) ** 2).sum(axis=1).max()))


def dual_value(points: np.ndarray, lam: np.ndarray) -> float:
    """Dualer Wert sum lam_i ||x_i||^2 - ||sum lam_i x_i||^2 für Gewichte im Simplex: eine untere Schranke für R*^2 (schwache Dualität)."""
    centre = lam @ points
    return float(lam @ (points * points).sum(axis=1) - centre @ centre)


def lower_radius(points: np.ndarray, lam: np.ndarray) -> float:
    """Untere Schranke für R*: Wurzel des dualen Werts (Rundungsreste unter null werden auf null gesetzt)."""
    return math.sqrt(max(dual_value(points, lam), 0.0))


def check_certificate(points: np.ndarray, lam: np.ndarray, c: np.ndarray, tol: float = 1e-9) -> list[str]:
    """Unabhängiger Prüfer: nennt jede verletzte Regel des Zertifikats (leere Liste = in Ordnung)."""
    problems = []
    if np.any(lam < -tol):
        problems.append("Gewichte negativ")
    if abs(float(lam.sum()) - 1.0) > tol:
        problems.append("Gewichte summieren sich nicht zu 1")
    if np.linalg.norm(lam @ points - c) > 1e-6 * (1.0 + np.linalg.norm(c)):
        problems.append("Mittelpunkt ist nicht der Gewichtsschwerpunkt")
    if lower_radius(points, lam) > upper_radius(points, c) * (1.0 + tol) + tol:
        problems.append("untere Schranke liegt über der oberen")
    return problems


# ----------------------------------------------------------------------------- Dualverfahren (Frank-Wolfe auf dem Simplex)

@dataclass(frozen=True, eq=False)
class DualRun:
    upper: np.ndarray    # upper[t] = obere Schranke im Zustand nach t Schritten (Länge steps + 1)
    lower: np.ndarray    # lower[t] = untere Schranke im Zustand nach t Schritten
    lam: np.ndarray      # Gewichte im Endzustand
    centre: np.ndarray   # Mittelpunkt im Endzustand (= lam @ points)


def fw_step(points: np.ndarray, sq: np.ndarray, lam: np.ndarray, c: np.ndarray, q_mean: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Ein Schritt: der fernste Punkt i* bekommt Gewicht gam (exakte Linienminimierung des Duals), der Mittelpunkt rückt auf ihn zu.

    `sq` sind die quadrierten Normen der Punkte, `q_mean` = lam @ sq. Gibt (lam, c, q_mean, bewegt) zurück; bewegt ist falsch, wenn der Mittelpunkt schon der fernste Punkt ist."""
    d2 = ((points - c) ** 2).sum(axis=1)
    i = int(np.argmax(d2))
    u = points[i] - c
    uu = float(u @ u)
    if uu == 0.0:
        return lam, c, q_mean, False
    gam = (sq[i] - q_mean - 2.0 * float(c @ u)) / (2.0 * uu)
    gam = min(max(gam, 0.0), 1.0)
    lam = lam * (1.0 - gam)
    lam[i] += gam
    return lam, c + gam * u, (1.0 - gam) * q_mean + gam * sq[i], True


def dual_run(points: np.ndarray, steps: int) -> DualRun:
    """Startet mit dem ganzen Gewicht auf dem ersten Punkt; speichert nach jedem Schritt beide Schranken."""
    n = len(points)
    if n < 1:
        raise ValueError("mindestens ein Punkt nötig")
    if steps < 0:
        raise ValueError("Schrittzahl darf nicht negativ sein")
    sq = (points * points).sum(axis=1)
    lam = np.zeros(n)
    lam[0] = 1.0
    c = points[0].astype(float).copy()
    q_mean = float(sq[0])
    upper, lower = np.empty(steps + 1), np.empty(steps + 1)
    for t in range(steps + 1):
        upper[t] = upper_radius(points, c)
        lower[t] = math.sqrt(max(q_mean - float(c @ c), 0.0))
        if t == steps:
            break
        lam, c, q_mean, moved = fw_step(points, sq, lam, c, q_mean)
        if not moved:
            upper[t + 1:], lower[t + 1:] = upper[t], lower[t]
            break
    return DualRun(upper, lower, lam, c)


def steps_to_gap(points: np.ndarray, eps: float, max_steps: int) -> int | None:
    """Erster Schritt t, ab dem (obere - untere Schranke) <= eps * obere Schranke gilt; None, wenn das in max_steps nicht gelingt."""
    sq = (points * points).sum(axis=1)
    lam = np.zeros(len(points))
    lam[0] = 1.0
    c = points[0].astype(float).copy()
    q_mean = float(sq[0])
    for t in range(max_steps + 1):
        up = upper_radius(points, c)
        if up - math.sqrt(max(q_mean - float(c @ c), 0.0)) <= eps * up:
            return t
        lam, c, q_mean, moved = fw_step(points, sq, lam, c, q_mean)
        if not moved:
            return t
    return None


# ----------------------------------------------------------------------------- exakte Kugel (Welzl) als Orakel in der App

def _circumsphere(support: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Mittelpunkt und Radius der kleinsten Kugel, die die Stützpunkte auf dem Rand hat und im Raum ihrer Differenzen liegt."""
    p = np.array(support)
    if len(p) == 1:
        return p[0].copy(), 0.0
    v = (p[1:] - p[0]).T
    s = (v * v).sum(axis=0)
    a = np.linalg.lstsq(v.T @ v, 0.5 * s, rcond=None)[0]
    c = p[0] + v @ a
    return c, float(np.linalg.norm(c - p[0]))


def smallest_ball(points: np.ndarray, order: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Exakte kleinste umschließende Kugel (Welzl, rekursiv). `order` ist eine Reihenfolge der Zeilen (Standard: wie gegeben)."""
    pts = np.asarray(points, dtype=float)
    n, d = pts.shape
    if n == 0:
        raise ValueError("leere Punktmenge")
    idx = list(range(n)) if order is None else [int(i) for i in order]

    def solve(count: int, support: list[int]) -> tuple[np.ndarray, float]:
        if count == 0 or len(support) == d + 1:
            return _circumsphere([pts[i] for i in support]) if support else (np.zeros(d), -1.0)
        p = idx[count - 1]
        c, r = solve(count - 1, support)
        if r >= 0 and np.linalg.norm(pts[p] - c) <= r * (1.0 + 1e-12) + 1e-12:
            return c, r
        return solve(count - 1, support + [p])

    c, r = solve(n, [])
    return c, max(r, 0.0)


# ----------------------------------------------------------------------------- nichtkonvexe Variante: Kugel um mindestens m der n Punkte

def trimmed_local(points: np.ndarray, m: int, c0: np.ndarray, max_iter: int = 200) -> tuple[np.ndarray, float]:
    """Wechselverfahren (lokale Suche): die m nächsten Punkte zum Mittelpunkt wählen, deren kleinste Kugel bilden, wiederholen, bis der Radius nicht mehr sinkt.

    Findet ein lokales Optimum ohne jeden Beweis, dass es das globale ist."""
    if not 1 <= m <= len(points):
        raise ValueError(f"m = {m} liegt außerhalb von 1 bis {len(points)}")
    c, radius = np.array(c0, dtype=float), math.inf
    for _ in range(max_iter):
        chosen = np.argsort(((points - c) ** 2).sum(axis=1), kind="stable")[:m]
        c_new, r_new = smallest_ball(points[chosen])
        if r_new >= radius - 1e-14:
            break
        c, radius = c_new, r_new
    return c, radius


def count_removals(n: int, r: int) -> int:
    """Zahl der Teilmengen, die der Beweis durch Aufzählen prüfen müsste: C(n, r) Möglichkeiten, r Punkte zu entfernen."""
    return math.comb(n, r)


def trimmed_exact(points: np.ndarray, r: int, limit: int) -> tuple[float, np.ndarray, int]:
    """Aufzählen: für jede Wahl von r zu entfernenden Punkten die kleinste Kugel der übrigen; gibt (Radius, Mittelpunkt, geprüfte Teilmengen) zurück.

    Bricht mit ValueError ab, wenn mehr als `limit` Teilmengen zu prüfen wären."""
    n = len(points)
    if not 0 <= r < n:
        raise ValueError(f"r = {r} liegt außerhalb von 0 bis {n - 1}")
    total = count_removals(n, r)
    if total > limit:
        raise ValueError(f"{total} Teilmengen, erlaubt sind {limit}")
    best_r, best_c = math.inf, None
    keep = np.ones(n, dtype=bool)
    for removed in itertools.combinations(range(n), r):
        keep[:] = True
        keep[list(removed)] = False
        c, radius = smallest_ball(points[keep])
        if radius < best_r:
            best_r, best_c = radius, c
    return best_r, best_c, total


def count_support_candidates(n: int, d: int) -> int:
    """Zahl der Kugeln, die durch höchstens d + 1 der n Punkte bestimmt sind (zwei bis d + 1 Stützpunkte): Sum_k C(n, k)."""
    return sum(math.comb(n, k) for k in range(2, d + 2))


def trimmed_exact_support(points: np.ndarray, m: int) -> tuple[float, np.ndarray]:   # (Radius, Mittelpunkt) wie trimmed_exact
    """Exakt über Stützpunkte: Die beste Kugel um mindestens m Punkte ist die kleinste Kugel ihrer Punkte, also durch höchstens d + 1 Punkte bestimmt.

    Jede solche Kugel wird gebildet; gewählt wird die kleinste, die mindestens m Punkte enthält. Anderer Rechenweg als das Aufzählen der entfernten Punkte;
    die Zahl der Kugeln wächst mit n^(d+1), in der Ebene also mäßig, in hoher Dimension explosionsartig."""
    pts = np.asarray(points, dtype=float)
    n, d = pts.shape
    if not 1 <= m <= n:
        raise ValueError(f"m = {m} liegt außerhalb von 1 bis {n}")
    if m == 1:
        return 0.0, pts[0].copy()
    best_r, best_c = math.inf, None
    for size in range(2, min(d + 1, n) + 1):
        for combo in itertools.combinations(range(n), size):
            c, radius = _circumsphere([pts[i] for i in combo])
            if radius >= best_r:
                continue
            if int((np.linalg.norm(pts - c, axis=1) <= radius * (1.0 + 1e-9) + 1e-12).sum()) >= m:
                best_r, best_c = radius, c
    return best_r, best_c


# ----------------------------------------------------------------------------- konvexe lokale Suche (Gegenstück zum Wechselverfahren)

def centre_walk(points: np.ndarray, c0: np.ndarray, steps: int) -> tuple[np.ndarray, float]:
    """Mittelpunkt-Verfahren (Bădoiu und Clarkson): im Schritt k um 1/(k+1) auf den fernsten Punkt zugehen. Startet bei einem beliebigen c0.

    Eine lokale Suche wie jede andere; nur die Konvexität garantiert, dass sie von jedem Start am selben Wert ankommt."""
    c = np.array(c0, dtype=float)
    for k in range(1, steps + 1):
        i = int(np.argmax(((points - c) ** 2).sum(axis=1)))
        c = c + (points[i] - c) / (k + 1)
    return c, upper_radius(points, c)
