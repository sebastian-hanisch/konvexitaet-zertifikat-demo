"""Szenario: Punktwolken in der Ebene für die kleinste umschließende Kugel (und ihre getrimmte Variante)."""

from dataclasses import dataclass

import numpy as np

import kvz_constants as C


@dataclass(frozen=True, eq=False)
class Instance:
    points: np.ndarray   # (n, d), Zeilen sind die Punkte

    @property
    def n(self) -> int:
        return len(self.points)


def make_cloud(kind: str, n: int, r: int, seed: int) -> Instance:
    """Deterministisch aus (kind, n, r, seed). `r` zählt nur bei „Haufen + Ausreißer“: so viele Punkte liegen verstreut rechts vom Haufen."""
    if kind not in C.CLOUDS:
        raise ValueError(f"unbekannte Wolke {kind!r}")
    rng = np.random.default_rng(seed)
    if kind == "Haufen + Ausreißer":
        outliers = min(r, n - 1)
        pts = np.vstack([rng.normal(0.0, 1.0, (n - outliers, 2)), rng.uniform(-8.0, 8.0, (outliers, 2)) + np.array([9.0, 0.0])])
    elif kind == "Zwei Haufen":
        half = n // 2
        pts = np.vstack([rng.normal([-3.0, 0.0], 0.8, (half, 2)), rng.normal([3.0, 0.0], 0.8, (n - half, 2))])
    else:
        pts = rng.normal(0.0, 1.0, (n, 2))
    return Instance(pts)


def make_gaussian(n: int, d: int, seed: int) -> np.ndarray:
    """Normalwolke in d Dimensionen für das Experiment zur Rate."""
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
