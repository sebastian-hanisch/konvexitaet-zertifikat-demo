"""Konvexität und Zertifikat – die Lücke als Beweis - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Reihe „Konvexe und konische Optimierung“: Am Beispiel der kleinsten umschließenden Kugel einer Punktwolke zeigt die Demo, dass
Konvexität eine zweite, duale Zahl liefert und die Lücke zwischen beiden den Fehler beweist; ihr nichtkonvexes Gegenstück (die Kugel um
mindestens m der n Punkte) hat kein solches Zertifikat. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import kvz_algorithm as A
import kvz_constants as C
from kvz_evaluation import (
    Settings,
    analyse,
    count_text,
    de,
    local_experiment,
    pct,
    playground,
    proof_experiment,
    rate_experiment,
    ratio_median,
    verdict_certificate,
    verdict_local,
    verdict_playground,
    verdict_proof,
    verdict_rate,
    verdict_trace,
)
from kvz_layout import dunkles_design, kopf_css
from kvz_pdf_export import build_pdf
from kvz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kvz_visualization import build_bounds, build_cloud, build_gap, build_local, build_proof, build_rate

st.set_page_config(page_title="Konvexität und Zertifikat – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _rate():
    return rate_experiment()


@st.cache_data(show_spinner=False)
def _local(settings):
    return local_experiment(settings)


@st.cache_data(show_spinner=False)
def _proof(settings):
    return proof_experiment(settings)


@st.cache_data(show_spinner=False)
def _trimmed(settings):
    a = analyse(settings)
    return A.trimmed_exact_support(a.inst.points, settings.m)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kvz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Konvexität und Zertifikat – die Lücke als Beweis")
st.markdown(
    """
**Was gibt Konvexität?** Nicht nur, dass jedes lokale Minimum global ist, sondern einen **Beweis zum Mitnehmen**: Zu jedem Zwischenergebnis gibt es eine zweite, **duale** Zahl,
und die **Lücke** zwischen beiden begrenzt den Fehler, ohne dass man die Lösung kennt. Die Demo zeigt das an der **kleinsten umschließenden Kugel** einer Punktwolke (Wurzel der Reihe
„Konvexe und konische Optimierung“) und stellt ihr die **getrimmte Kugel** gegenüber, die mindestens m der n Punkte enthalten soll: dort fehlt die Konvexität, und mit ihr der billige Beweis.

Weiter unten: **vier Experimente** (**🔬**), eine **🎲 Spielwiese**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/beweis-zum-mitnehmen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Die Demo sucht den **kleinsten Kreis** (in mehr Dimensionen: die kleinste Kugel), der alle Punkte einer Wolke umschließt, etwa als Standort mit kürzestem Weg zum fernsten Kunden. Zu jedem Zwischenergebnis rechnet sie zwei Zahlen aus: eine **obere Schranke** (der Kreis, den man gerade hat) und eine **untere Schranke** (eine Länge, die der beste Kreis nicht unterschreiten kann, welche Gewichte man auch auf die Punkte verteilt). Die **Lücke** zwischen beiden begrenzt den Fehler, ohne dass man die beste Lösung kennt: der **Beweis zum Mitnehmen**.

Beim Gegenstück, der **getrimmten Kugel**, die nur die meisten Punkte umschließen muss, fehlt die Konvexität und mit ihr dieser Beweis; das Wechselverfahren trifft das Optimum nur in einem Teil der Läufe. Der Haken: Das Zertifikat überschätzt den wahren Fehler in den Messungen etwa um das Doppelte.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/beweis-zum-mitnehmen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Zertifikat", expanded=True):
    st.markdown(
        """
1. **Obere Schranke (primal):** Jeder Mittelpunkt c gibt eine Kugel, die alle Punkte enthält; ihr Radius ist der Abstand zum fernsten Punkt. Der kleinste Radius R* ist höchstens so groß.
2. **Untere Schranke (dual):** Jede Verteilung von Gewichten auf die Punkte (alle nicht negativ, Summe 1) gibt eine Zahl, die R* höchstens erreicht: die Streuung der Punkte um ihren Gewichtsschwerpunkt. Das gilt für **jede** Verteilung.
3. **Das Verfahren:** In jedem Schritt bekommt der **fernste** Punkt mehr Gewicht, der Mittelpunkt (der Gewichtsschwerpunkt) rückt auf ihn zu. Beide Schranken laufen aufeinander zu.
4. **Warum das ein Beweis ist:** Liegen beide Schranken nahe beieinander, steckt R* dazwischen, auf welchem Weg man dorthin kam. Die Lücke ist die Garantie.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Punktwolke**")
    cloud = st.selectbox("Art der Wolke", C.CLOUDS, key="cloud_select", help="Normalverteilt, ein Haufen mit Ausreißern oder zwei Haufen.")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    r = st.slider("Auszusortierende Punkte r", *bounds("r_slider"), key="r_slider",
                  help="Die getrimmte Kugel muss nur n − r Punkte enthalten. Bei „Haufen + Ausreißer“ liegen so viele Punkte verstreut rechts vom Haufen.")
    st.markdown("**Dualverfahren**")
    steps = st.slider("Schritte", *bounds("steps_slider"), key="steps_slider", help="Wie viele Schritte das Verfahren macht, bevor die Schranken gelesen werden.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"cloud_select": cloud, "n_slider": int(n), "r_slider": int(r), "steps_slider": int(steps), "seed_input": int(seed)})

settings = Settings(cloud, int(n), int(r), int(steps), int(seed))
a = _analysis(settings)
if a.problems:   # Prüfer (rechnet die Schranken neu), nicht nur im Test: lieber eine Fehlermeldung als eine stille falsche Zahl
    st.error("Das Zertifikat besteht den Prüfer nicht: " + "; ".join(a.problems))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Obere Schranke", de(a.upper, 3), help="Radius der Kugel um den Mittelpunkt c: sie enthält alle Punkte.")
m2.metric("Untere Schranke (dual)", de(a.lower, 3), help="Der duale Wert der aktuellen Gewichte: kleiner kann der kleinste Radius nicht sein.")
m3.metric("Bewiesene Lücke", pct(a.gap_rel), help="Obere minus untere Schranke, in Anteilen der oberen. Absolut liegt die obere Schranke höchstens so weit daneben.")
m4.metric("Wahrer Radius (exakt)", de(a.exact_r, 3), help="Zum Vergleich exakt berechnet (Welzl). Das Zertifikat braucht diese Zahl nicht.")
st.plotly_chart(
    build_cloud(a.inst.points, a.run.centre, a.upper, a.run.lam, a.exact_centre, a.exact_r),
    width="stretch", key="main_chart",
)
st.caption("Rot: die Kugel, die das Verfahren gerade vorschlägt. Blau gestrichelt: der wahre kleinste Radius. Große Punkte tragen Gewicht im Dual: Sie bestimmen die untere Schranke.")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kleinste-kugel-zertifikat.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie gut ist die Lösung, ohne sie zu kennen?")
verdict(verdict_certificate(a.upper, a.lower, a.exact_r, settings.steps))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Zertifikat im Verlauf")
st.caption("Beide Schranken über die Schritte der eingestellten Wolke. Links die Radien, rechts die relative Lücke (bewiesen) und der wahre Fehler der oberen Schranke (nur mit bekannter Lösung).")
c1, c2 = st.columns(2)
if settings.steps >= 2:
    c1.plotly_chart(build_bounds(a.run, a.exact_r), width="stretch", key="bounds_chart")
    c2.plotly_chart(build_gap(a.run, a.exact_r), width="stretch", key="gap_chart")
else:
    st.info("Mit einem Schritt gibt es noch keinen Verlauf; mehr Schritte einstellen.")
verdict(verdict_trace(ratio_median(a)))

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie viele Schritte kostet die Genauigkeit?")
st.caption(f"Normalverteilte Wolken in 2 bis 100 Dimensionen, Median über {len(C.SWEEP_SEEDS)} feste Wolken (SWEEP_SEEDS); gezählt werden die Schritte, bis die relative Lücke unter ε fällt.")
rate_rows = _rate()
st.plotly_chart(build_rate(rate_rows), width="stretch", key="rate_chart")
verdict(verdict_rate(rate_rows))

st.markdown("---")

st.subheader("🔬 Experiment 3: Lokal gleich global?")
st.caption(f"{C.START_COUNT} Zufallsstarts für den Mittelpunkt: das konvexe Mittelpunkt-Verfahren ({C.START_STEPS} Schritte) gegen das Wechselverfahren der getrimmten Kugel "
           f"(die {settings.m} nächsten Punkte wählen, deren Kugel bilden, wiederholen). Rechnet einige Sekunden.")
if st.button("🔬 Experiment 3 rechnen", key="local_run"):
    st.session_state["local_res"] = (settings, _local(settings))
res = st.session_state.get("local_res")
if res and res[0] == settings:
    st.plotly_chart(build_local(res[1]), width="stretch", key="local_chart")
    verdict(verdict_local(res[1], settings.m))
else:
    st.info("Noch nicht gerechnet für diese Einstellungen.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Was kostet der Beweis ohne Konvexität?")
st.caption(f"Das Optimum der getrimmten Kugel (mindestens {settings.m} von {settings.n} Punkten) beweist man, indem man alle Möglichkeiten prüft. "
           "Zwei Wege: alle Teilmengen mit r entfernten Punkten, oder alle Kugeln, die durch höchstens d + 1 Punkte bestimmt sind.")
proof_rows = _proof(settings)
st.plotly_chart(build_proof(proof_rows, settings.r), width="stretch", key="proof_chart")
verdict(verdict_proof(proof_rows, settings.r, settings.n, C.BRUTE_FORCE_LIMIT))
best_r, best_c = _trimmed(settings)
subsets = A.count_removals(settings.n, settings.r)
st.markdown(f"**Exakt über Stützpunkte** (in der Ebene schnell): kleinste Kugel um mindestens {settings.m} Punkte hat Radius **{de(best_r, 3)}**, "
            f"gegenüber {de(a.exact_r, 3)} für alle {settings.n} Punkte.")
if subsets <= C.BRUTE_FORCE_LIMIT:
    if st.button(f"🔬 Alle {count_text(subsets)} Teilmengen live prüfen", key="brute_run"):
        st.session_state["brute_res"] = (settings, A.trimmed_exact(a.inst.points, settings.r, C.BRUTE_FORCE_LIMIT))
    brute = st.session_state.get("brute_res")
    if brute and brute[0] == settings:
        agree = abs(brute[1][0] - best_r) <= 1e-9 * max(best_r, 1.0)
        (st.success if agree else st.error)(f"Aufzählung über {brute[1][2]} Teilmengen: Radius {de(brute[1][0], 3)}; "
                                            + ("gleich dem Ergebnis über Stützpunkte." if agree else "weicht vom Ergebnis über Stützpunkte ab: Das darf nicht passieren."))
else:
    st.info(f"{count_text(subsets)} Teilmengen sind mehr als die {C.BRUTE_FORCE_LIMIT} der Live-Aufzählung; mit weniger Punkten oder weniger auszusortierenden Punkten lässt sie sich ausführen.")
st.plotly_chart(build_cloud(a.inst.points, trimmed=(best_c, best_r), exact_centre=a.exact_centre, exact_r=a.exact_r), width="stretch", key="trim_chart")
st.caption("Blau gestrichelt: Kugel um alle Punkte. Grün gepunktet: kleinste Kugel um mindestens n − r Punkte.")

st.markdown("---")

st.subheader("🎲 Spielwiese: Schwache Dualität zum Anfassen")
st.caption("Setzen Sie einen Mittelpunkt von Hand und würfeln Sie Gewichte. Die Reihenfolge untere Schranke ≤ wahrer Radius ≤ obere Schranke gilt immer, so schlecht beide auch gewählt sind.")
p1, p2, p3, p4 = st.columns(4)
cx = p1.slider("Mittelpunkt x", -20.0, 20.0, 0.0, 0.1, key="play_cx")
cy = p2.slider("Mittelpunkt y", -20.0, 20.0, 0.0, 0.1, key="play_cy")
alpha = p3.slider("Gleichmäßigkeit der Gewichte", 0.05, 5.0, 1.0, 0.05, key="play_alpha", help="Klein: wenige Punkte bekommen fast alles. Groß: alle ähnlich.")
dice = p4.number_input("Würfel", 0, 999, 0, step=1, key="play_dice")
lam = np.random.default_rng(int(dice)).dirichlet(np.full(settings.n, float(alpha)))
vals = playground(a.inst.points, np.array([cx, cy]), lam)
st.plotly_chart(build_cloud(a.inst.points, np.array([cx, cy]), vals["upper"], lam, a.exact_centre, a.exact_r), width="stretch", key="play_chart")
verdict(verdict_playground(vals))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur die euklidische Kugel** | Andere Formen (Würfel, Ellipsoid) sind andere Kegelprogramme; die Demo rechnet nur die Kugel. | Stück 2 der Reihe: Kegelform |
| **Die Dualität hat keine Lücke** | Bei der Kugel gilt starke Dualität (es gibt einen Punkt im Inneren). Bei anderen konvexen Problemen kann die Lücke trotz Konvexität bleiben. | Stück 4: Kegel-Dualität |
| **Genauigkeit kostet Schritte** | Das Verfahren braucht bei Normalwolken etwa 0,5 bis 0,9/ε Schritte für die relative Lücke ε (gemessen bis ε = 10⁻³); für 10⁻⁶ ergab eine Nachmessung 0,6 bis 1,0 Millionen Schritte, in der App nicht gerechnet. Schnellere Verfahren folgen. | Stück 5 und 6: Innere Punkte |
| **Die getrimmte Kugel ist nur ein nichtkonvexes Beispiel** | Das Wechselverfahren ist eine von vielen lokalen Suchen; bessere Heuristiken treffen öfter, beweisen aber nichts. In der Ebene gibt es einen exakten Weg über Stützpunkte, in hoher Dimension nicht billig. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/beweis-zum-mitnehmen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Punkte $x_1, \dots, x_n$ im $\mathbb{R}^d$. Gesucht ist die kleinste Kugel, die alle enthält. Eine obere Schranke liefert jeder Mittelpunkt $c$, eine untere jede Verteilung von Gewichten $\lambda$ auf die Punkte.

| Symbol | Bedeutung |
|---|---|
| $x_i,\ n,\ d$ | Punkt $i$, Zahl der Punkte und Dimension (in der Ebene $d = 2$) |
| $c,\ R$ | Mittelpunkt und Radius einer Kugel; $R^*$ ist der kleinste mögliche Radius |
| $\lVert \cdot \rVert$ | euklidischer Abstand (Norm) |
| $U(c)$ | obere Schranke: Abstand von $c$ zum fernsten Punkt |
| $\lambda,\ \lambda_i$ | Gewichte auf den Punkten (nicht negativ, Summe 1) |
| $\bar x$ | Gewichtsschwerpunkt $\sum_i \lambda_i x_i$ |
| $D(\lambda),\ \sqrt{D(\lambda)}$ | gewichtete Streuung der Punkte um $\bar x$ und daraus die untere Schranke |
| $\gamma,\ e_i$ | Schrittweite und Einheitsvektor (das ganze Gewicht auf Punkt $i$) |
| $m,\ r,\ S$ | mindestens zu umschließende Punktzahl, $r = n - m$ auszusortierende Punkte und die gewählte Auswahl von $m$ Punkten |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/beweis-zum-mitnehmen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Problem.** Gegeben Punkte $x_1, \dots, x_n \in \mathbb{R}^d$. Gesucht ist die kleinste Kugel, die alle enthält:
$R^* = \min_{c,R} R$ unter $\lVert x_i - c \rVert \le R$ für alle $i$. Das Problem ist konvex (jede Nebenbedingung ist eine Norm, also konvex).

**Obere Schranke.** Für jedes $c$ ist $U(c) = \max_i \lVert x_i - c \rVert \ge R^*$.

**Untere Schranke (schwache Dualität).** Für Gewichte $\lambda \ge 0$ mit $\sum_i \lambda_i = 1$ sei $\bar x = \sum_i \lambda_i x_i$ und
$D(\lambda) = \sum_i \lambda_i \lVert x_i \rVert^2 - \lVert \bar x \rVert^2 = \sum_i \lambda_i \lVert x_i - \bar x \rVert^2$. Für die optimale Kugel $(c^*, R^*)$ gilt
$R^{*2} \ge \sum_i \lambda_i \lVert x_i - c^* \rVert^2 \ge \sum_i \lambda_i \lVert x_i - \bar x \rVert^2 = D(\lambda)$, denn der Gewichtsschwerpunkt minimiert die gewichtete Streuung. Also $\sqrt{D(\lambda)} \le R^* \le U(c)$.

**Starke Dualität.** Für diese Aufgabe gilt $\max_\lambda D(\lambda) = R^{*2}$ (die Aufgabe hat einen strikt zulässigen Punkt, die Slater-Bedingung ist erfüllt).

**Verfahren (Frank-Wolfe auf dem Dual).** Start $\lambda = e_1$. Im Schritt sei $c = \sum_i \lambda_i x_i$ und $i^*$ der fernste Punkt; setze $\lambda \leftarrow (1-\gamma)\lambda + \gamma e_{i^*}$ mit
exakter Linienminimierung für $\gamma \in [0,1]$. Dann ist die obere Schranke $U(c)$ und die untere $\sqrt{D(\lambda)}$.

**Getrimmte Kugel.** Kleinste Kugel, die mindestens $m$ der $n$ Punkte enthält: $\min_{S,\, |S| = m} R^*(S)$. Die Zielfunktion ist nicht konvex in den Auswahlvariablen; es gibt keine billige duale Schranke, die im Allgemeinen scharf ist.
Exakt: für jede Wahl von $r = n - m$ entfernten Punkten die kleinste Kugel der übrigen ($\binom{n}{r}$ Kugeln), oder alle Kugeln durch höchstens $d + 1$ Punkte ($\sum_{k=2}^{d+1} \binom{n}{k}$ Kugeln).

**Literatur.** E. Welzl: *Smallest enclosing disks (balls and ellipsoids)*, in: H. Maurer (Hrsg.), New Results and New Trends in Computer Science, Lecture Notes in Computer Science 555, Springer 1991, S. 359–370 (exakte kleinste Kugel in erwarteter Linearzeit bei zufälliger Reihenfolge der Punkte).
M. Bădoiu und K. L. Clarkson: *Smaller core-sets for balls*, Proceedings of the 14th ACM-SIAM Symposium on Discrete Algorithms (SODA) 2003, S. 801–802 (Kernmengen der Größe $O(1/\varepsilon)$, unabhängig von der Dimension; das Mittelpunkt-Verfahren der Demo geht auf diese Art von Schritten zurück; die Rate der Demo ist daraus nicht abgeleitet).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
