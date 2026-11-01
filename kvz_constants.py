"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

CLOUDS = ("Normalwolke", "Haufen + Ausreißer", "Zwei Haufen")
DEFAULT_CLOUD = CLOUDS[0]
N_MIN, N_MAX, DEFAULT_N = 8, 40, 20
R_MIN, R_MAX, DEFAULT_R = 1, 6, 4          # r = Zahl der auszusortierenden Punkte; N_MIN - R_MAX >= 2: die getrimmte Kugel hat immer mindestens zwei Punkte
STEPS_MIN, STEPS_MAX, DEFAULT_STEPS = 1, 2000, 200   # Schritte des Dualverfahrens, O(n d) je Schritt
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100005))  # feste Wolken für alle Sweeps: nie ein einzelner Seed als Befund
RATE_EPS = (0.1, 0.01, 0.001)              # relative Lücke, bei der gezählt wird (1e-3 braucht etwa 700 Schritte)
RATE_SIZES = ((50, 2), (200, 2), (200, 10), (200, 100), (1000, 10))   # (Punkte n, Dimension d); O(n d) je Schritt, höchstens 1000 x 10
RATE_MAX_STEPS = 5000
START_COUNT = 100                          # Zufallsstarts im Experiment „lokal gleich global“
START_STEPS = 2000                         # Schritte des Mittelpunkt-Verfahrens je Start
BRUTE_FORCE_LIMIT = 2000                   # so viele Teilmengen darf die Aufzählung in der App höchstens prüfen (einige ms je Teilmenge, also einige Sekunden)
PROOF_DIMS = (2, 3, 5, 10)                 # Dimensionen für die Zahl der Stützpunkt-Kugeln
PROOF_SIZES = (10, 15, 20, 25, 30, 35, 40)
TIMING_SAMPLES = 20                        # Teilmengen für die Zeitmessung je Größe

COLOR_UPPER = "#c0504d"                    # obere Schranke (Kugel um den Mittelpunkt c)
COLOR_LOWER = "#2F6B65"                    # untere Schranke (dualer Wert)
COLOR_EXACT = "#4c78a8"
COLOR_POINT = "rgba(150,150,150,0.75)"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall: Normalwolke": {"cloud": DEFAULT_CLOUD, "n": DEFAULT_N, "r": DEFAULT_R, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Ein Haufen, vier Ausreißer": {"cloud": "Haufen + Ausreißer", "n": DEFAULT_N, "r": 4, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Zwei Haufen": {"cloud": "Zwei Haufen", "n": DEFAULT_N, "r": 4, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Wenige Schritte (5)": {"cloud": DEFAULT_CLOUD, "n": DEFAULT_N, "r": DEFAULT_R, "steps": 5, "seed": DEFAULT_SEED},
    "Beweis wird unbezahlbar": {"cloud": DEFAULT_CLOUD, "n": N_MAX, "r": R_MAX, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: Normalwolke": "20 Punkte aus einer Normalverteilung, 200 Schritte des Dualverfahrens, vier Punkte werden für die getrimmte Kugel aussortiert.",
    "Ein Haufen, vier Ausreißer": "16 Punkte im Haufen und vier weit rechts: die Kugel um alle Punkte wird riesig, die getrimmte klein, und das lokale Verfahren verläuft sich oft.",
    "Zwei Haufen": "Zwei gleich große Haufen: zwei Teillösungen konkurrieren, wenn Punkte aussortiert werden dürfen.",
    "Wenige Schritte (5)": "Nur fünf Schritte: die Lücke ist noch groß, aber der Beweis gilt trotzdem.",
    "Beweis wird unbezahlbar": "40 Punkte, 6 aussortieren: knapp 3,8 Millionen Teilmengen, zu viele für die Aufzählung in der App.",
}
