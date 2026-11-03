"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

import math
import time
from dataclasses import dataclass
from statistics import median

import numpy as np

import kvz_algorithm as A
import kvz_constants as C
from kvz_scenario import Instance, make_cloud, make_gaussian


@dataclass(frozen=True)
class Settings:
    cloud: str = C.DEFAULT_CLOUD
    n: int = C.DEFAULT_N
    r: int = C.DEFAULT_R
    steps: int = C.DEFAULT_STEPS
    seed: int = C.DEFAULT_SEED

    @property
    def m(self) -> int:
        """Zahl der Punkte, die die getrimmte Kugel mindestens enthalten muss."""
        return self.n - self.r


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    inst: Instance
    exact_centre: np.ndarray
    exact_r: float
    run: A.DualRun
    problems: list

    @property
    def upper(self) -> float:
        return float(self.run.upper[-1])

    @property
    def lower(self) -> float:
        return float(self.run.lower[-1])

    @property
    def gap_rel(self) -> float:
        """Lücke in Anteilen der oberen Schranke: der größte Fehler, den das Zertifikat noch zulässt."""
        return 0.0 if self.upper <= 0 else (self.upper - self.lower) / self.upper


# ----------------------------------------------------------------------------- Formatierung

_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(frac: float) -> str:
    """Anteil als Prozentzahl: ab 1 % mit einer Nachkommastelle, darunter mit zwei."""
    p = 100.0 * frac
    return de(p, 1 if p >= 1.0 else 2) + " %"


def count_text(n: float) -> str:
    """Große Zahlen in Worten: bis 9999 ausgeschrieben, darüber Tausend, Millionen, Milliarden, Billionen, darüber Zehnerpotenz."""
    if n < 10_000:
        return str(int(n))
    for scale, name in ((1e3, "Tausend"), (1e6, "Millionen"), (1e9, "Milliarden"), (1e12, "Billionen")):
        if n < scale * 1000:
            value = n / scale
            return f"{de(value, 0 if value >= 100 else 1)} {name}"
    e = int(math.floor(math.log10(n)))
    return f"{de(n / 10 ** e, 1)}·10{str(e).translate(_SUPER)}"


def time_text(seconds: float) -> str:
    """Dauer in der passenden Einheit; ab einem Jahr in Jahren."""
    for limit, unit, div in ((60, "Sekunden", 1), (3600, "Minuten", 60), (86400, "Stunden", 3600), (365.25 * 86400, "Tagen", 86400)):
        if seconds < limit:
            return f"{de(seconds / div, 0 if seconds / div >= 100 else 1)} {unit}"
    years = seconds / (365.25 * 86400)
    return f"{de(years, 1)} Jahren" if years < 10_000 else f"{count_text(years)} Jahren"


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def analyse(settings: Settings) -> Analysis:
    inst = make_cloud(settings.cloud, settings.n, settings.r, settings.seed)
    centre, radius = A.smallest_ball(inst.points, np.random.default_rng(settings.seed).permutation(inst.n))
    run = A.dual_run(inst.points, settings.steps)
    problems = A.check_certificate(inst.points, run.lam, run.centre)
    return Analysis(settings, inst, centre, radius, run, problems)


def error_gap_ratios(upper: np.ndarray, lower: np.ndarray, exact_r: float, min_gap: float = 1e-9) -> np.ndarray:
    """(obere Schranke - wahrer Radius) / Lücke über den Verlauf; liegt nach schwacher Dualität zwischen 0 und 1. Zustände ohne Lücke werden übergangen."""
    gap = upper - lower
    mask = gap > min_gap
    return (upper[mask] - exact_r) / gap[mask]


def ratio_median(a: Analysis) -> float:
    ratios = error_gap_ratios(a.run.upper, a.run.lower, a.exact_r)
    return float("nan") if len(ratios) == 0 else float(np.median(ratios))


# ----------------------------------------------------------------------------- Experiment: Schritte bis zur Lücke

def rate_experiment(sizes=C.RATE_SIZES, eps_list=C.RATE_EPS, seeds=C.SWEEP_SEEDS, max_steps: int = C.RATE_MAX_STEPS) -> list[dict]:
    """Median über feste Wolken: Schritte, bis die relative Lücke unter eps fällt, und das Produkt Schritte mal eps (nan, wenn max_steps nicht reichen)."""
    rows = []
    for n, d in sizes:
        counts = {eps: [] for eps in eps_list}
        for seed in seeds:
            pts = make_gaussian(n, d, seed)
            for eps in eps_list:
                t = A.steps_to_gap(pts, eps, max_steps)
                counts[eps].append(float("nan") if t is None else float(t))
        rows.append({"n": n, "d": d, "steps": {eps: float(np.nanmedian(v)) if not np.all(np.isnan(v)) else float("nan") for eps, v in counts.items()}})
    for row in rows:
        row["product"] = {eps: row["steps"][eps] * eps for eps in eps_list}
    return rows


# ----------------------------------------------------------------------------- Experiment: lokal gleich global?

def local_experiment(settings: Settings, starts: int = C.START_COUNT, walk_steps: int = C.START_STEPS) -> dict:
    """Von Zufallsstarts aus: konvexes Mittelpunkt-Verfahren (Ziel: kleinste Kugel um alle) und Wechselverfahren (Ziel: kleinste Kugel um mindestens m Punkte)."""
    a = analyse(settings)
    pts = a.inst.points
    rng = np.random.default_rng([settings.seed, 7])
    lo, hi = pts.min(axis=0) - 2.0, pts.max(axis=0) + 2.0
    walk_err, trim_vals = [], []
    for _ in range(starts):
        c0 = rng.uniform(lo, hi)
        walk_err.append(A.centre_walk(pts, c0, walk_steps)[1] / a.exact_r - 1.0)
        trim_vals.append(A.trimmed_local(pts, settings.m, c0)[1])
    best_r, _ = A.trimmed_exact_support(pts, settings.m)
    walk_err, trim_vals = np.array(walk_err), np.array(trim_vals)
    return {
        "starts": starts,
        "walk_err": walk_err,
        "walk_max_err": float(walk_err.max()),
        "walk_share_1pct": float((walk_err <= 0.01).mean()),
        "trim_best": float(best_r),
        "trim_values": trim_vals,
        "trim_share_hit": float((np.abs(trim_vals - best_r) <= 1e-9 * max(best_r, 1.0)).mean()),
        "trim_worst_over": float(trim_vals.max() / best_r - 1.0) if best_r > 0 else 0.0,
        "trim_distinct": int(len(np.unique(np.round(trim_vals, 6)))),
    }


# ----------------------------------------------------------------------------- Experiment: was kostet der Beweis ohne Konvexität?

def time_per_subset(settings: Settings, n: int, samples: int = C.TIMING_SAMPLES) -> float:
    """Gemessene Sekunden, um für eine Teilmenge mit n - r Punkten die kleinste Kugel zu bestimmen (Mittel über `samples` zufällige Teilmengen)."""
    pts = make_cloud(settings.cloud, n, settings.r, settings.seed).points
    rng = np.random.default_rng([settings.seed, n])
    subsets = [np.sort(rng.permutation(n)[: n - settings.r]) for _ in range(samples)]
    t0 = time.perf_counter()
    for keep in subsets:
        A.smallest_ball(pts[keep])
    return (time.perf_counter() - t0) / samples


def proof_experiment(settings: Settings, sizes=C.PROOF_SIZES, dims=C.PROOF_DIMS) -> list[dict]:
    """Je Größe n: Zahl der Teilmengen C(n, r), gemessene Zeit je Teilmenge, Hochrechnung, und die Zahl der Stützpunkt-Kugeln für mehrere Dimensionen."""
    rows = []
    for n in sizes:
        count = A.count_removals(n, settings.r)
        per = time_per_subset(settings, n)
        rows.append({"n": n, "subsets": count, "per_subset": per, "seconds": count * per,
                     "support": {d: A.count_support_candidates(n, d) for d in dims}})
    return rows


# ----------------------------------------------------------------------------- Spielwiese: schwache Dualität zum Anfassen

def playground(points: np.ndarray, centre: np.ndarray, lam: np.ndarray) -> dict:
    """Obere Schranke aus einem beliebigen Mittelpunkt, untere aus beliebigen Gewichten im Simplex, dazu der wahre Radius."""
    exact_r = A.smallest_ball(points)[1]
    return {"upper": A.upper_radius(points, centre), "lower": A.lower_radius(points, lam), "exact": exact_r}


# ----------------------------------------------------------------------------- Verdikte (rein)

def verdict_certificate(upper: float, lower: float, exact_r: float, steps: int) -> tuple[str, str]:
    """Kernfrage: Wie gut ist die Lösung, ohne sie zu kennen? Die Lücke begrenzt den Fehler; der wahre Fehler steht zum Vergleich daneben."""
    gap = 0.0 if upper <= 0 else (upper - lower) / upper
    true_err = 0.0 if exact_r <= 0 else (upper - exact_r) / exact_r
    base = (f"Nach {steps} Schritten liegt der kleinste Radius **zwischen {de(lower, 3)} und {de(upper, 3)}**. Die Lücke von {pct(gap)} (Anteil der oberen Schranke) ist bewiesen: "
            f"Absolut liegt die obere Schranke höchstens so weit daneben, wie obere und untere Schranke auseinanderliegen. "
            f"Der wahre Radius ist {de(exact_r, 3)}, der Fehler also {pct(true_err)} (Anteil des wahren Radius, eine andere Bezugsgröße als bei der Lücke).")
    if gap <= 0.01:
        return "success", base
    if gap <= 0.10:
        return "info", base
    return "warning", base + " Die Lücke ist noch groß, aber sie ist ehrlich: Das Zertifikat sagt genau, wie wenig es schon beweist."


def verdict_trace(ratio: float) -> tuple[str, str]:
    """Experiment 1: Der wahre Fehler im Vergleich zur Lücke (Anteil zwischen 0 und 1)."""
    if math.isnan(ratio):
        return "info", "Die Lücke ist schon null; es gibt keinen Verlauf zu vergleichen."
    return "info", (f"Im Median über den Verlauf ist der wahre Fehler der oberen Schranke das **{de(ratio, 2)}-Fache der Lücke**. "
                    f"Das Zertifikat überschätzt den Fehler also um den Faktor {de(1.0 / ratio, 1) if ratio > 0 else 'unendlich'}, bleibt aber immer auf der sicheren Seite.")


def verdict_rate(rows: list[dict]) -> tuple[str, str]:
    """Experiment 2: Schritte bis zur relativen Lücke eps, über Punktzahl und Dimension."""
    smallest = min(eps for eps in rows[0]["steps"])
    products = [r["product"][smallest] for r in rows if not math.isnan(r["product"][smallest])]
    lo, hi = min(products), max(products)
    big = max(rows, key=lambda r: r["n"] * r["d"])
    small = min(rows, key=lambda r: r["n"] * r["d"])
    return "info", (f"Bis zur relativen Lücke {pct(smallest)} brauchen alle Wolken **{count_text(min(r['steps'][smallest] for r in rows))} bis {count_text(max(r['steps'][smallest] for r in rows))} Schritte**, "
                    f"das ist {de(lo, 2)} bis {de(hi, 2)} mal 1/ε, obwohl n·d von {small['n'] * small['d']} bis {big['n'] * big['d']} reicht. "
                    "Bei diesen Normalwolken kostet zehnmal mehr Genauigkeit etwa zehnmal mehr Schritte: Die Rate hängt von der Lücke ab, nicht von der Größe.")


def verdict_local(res: dict, m: int) -> tuple[str, str]:
    """Experiment 3: Zufallsstarts, konvex gegen nichtkonvex."""
    base = (f"**Konvex:** {pct(res['walk_share_1pct'])} der {res['starts']} Starts enden höchstens 1 % über dem wahren Radius (schlechtester Start {pct(res['walk_max_err'])}). "
            f"**Getrimmt (mindestens {m} Punkte):** nur {pct(res['trim_share_hit'])} der Starts treffen das Optimum {de(res['trim_best'], 3)}, "
            f"der schlechteste Start liegt {pct(res['trim_worst_over'])} darüber, und es gibt {res['trim_distinct']} verschiedene Endwerte.")
    if res["trim_share_hit"] < 0.5:
        return "warning", base + " Ohne Konvexität weiß das lokale Verfahren nicht, ob es angekommen ist."
    return "info", base + " Hier treffen die meisten Starts, aber ein Beweis, dass es das Optimum ist, fehlt trotzdem."


def verdict_proof(rows: list[dict], r: int, n: int, limit: int) -> tuple[str, str]:
    """Experiment 4: Aufwand des Beweises durch Aufzählen und durch Stützpunkte."""
    row = min(rows, key=lambda x: abs(x["n"] - n))
    d_hi = max(row["support"])
    return "warning", (f"Alle Teilmengen zu prüfen, die nach dem Entfernen von {r} Punkten übrig bleiben, heißt bei n = {row['n']} Punkten {count_text(row['subsets'])} Kugeln, "
                       f"bei {de(row['per_subset'] * 1000, 1)} ms je Kugel auf diesem Server etwa {time_text(row['seconds'])}. "
                       f"In der Ebene geht es klüger (Kugeln durch höchstens drei Punkte: {count_text(row['support'][2])}), aber in {d_hi} Dimensionen wären es {count_text(row['support'][d_hi])} "
                       f"Kugeln. Der Beweis wächst mit der Dimension; die Lücke des konvexen Falls kostet in jeder Dimension nur O(n·d) je Schritt.")


def verdict_playground(vals: dict) -> tuple[str, str]:
    """Spielwiese: beliebiger Mittelpunkt und beliebige Gewichte; die Reihenfolge untere ≤ wahr ≤ obere gilt immer."""
    lo, ex, up = vals["lower"], vals["exact"], vals["upper"]
    tol = 1e-9 * max(ex, 1.0)
    if lo <= ex + tol and ex <= up + tol:
        return "success", (f"{de(lo, 3)} ≤ {de(ex, 3)} ≤ {de(up, 3)}: Ihre Gewichte geben eine untere, Ihr Mittelpunkt eine obere Schranke. "
                           f"Der wahre Radius liegt dazwischen, egal wie schlecht beide gewählt sind. Gleich groß werden sie nur im Optimum.")
    return "error", "Die Reihenfolge ist verletzt: Das darf nicht passieren, die Berechnung ist falsch."
