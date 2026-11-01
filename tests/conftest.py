"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Vier Punkte (0,0), (4,0), (0,3), (1,1). Handrechnung: Die Hypotenuse von (4,0) nach (0,3) hat Länge 5, also Kugel um (2; 1,5) mit Radius 2,5; (1,1) liegt innen.
    Dual: Gewichte (0; 0,5; 0,5; 0) geben 0,5·16 + 0,5·9 − |(2; 1,5)|² = 12,5 − 6,25 = 6,25 = 2,5²."""
    return np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [1.0, 1.0]])


@pytest.fixture
def trap():
    """Vier Punkte (0,0), (1,0), (0,1), (10,0). Getrimmt mit m = 3 (einen Punkt aussortieren): Optimum um die ersten drei, Radius √2/2 bei (0,5; 0,5).
    Das Wechselverfahren aus c = (10, 0) wählt (10,0), (1,0), (0,0), hat Radius 5 und bleibt dort hängen."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
