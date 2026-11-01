# Konvexität und Zertifikat – die Lücke als Beweis – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-konvexitaet-zertifikat-demo.streamlit.app/)** -->

Wurzel der **Konzepte-Reihe „Konvexe und konische Optimierung“** (Stück 1 von 16) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Primal-Dual-Löser, dessen Kegel von Stück zu Stück wächst. Dieses Stück legt den Grund: Was gibt Konvexität, wenn man sie ernst nimmt? Die Reihe baut darauf die Kegelform (LP, SOCP, SDP), Kegel-Dualität, Innere-Punkte-Verfahren, Splitting-Verfahren und Relaxationen auf; die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Was gibt Konvexität außer „jedes lokale Minimum ist global“?** Einen **Beweis zum Mitnehmen**. Zu jedem Zwischenergebnis gibt es eine zweite, duale Zahl, und die **Lücke** zwischen beiden begrenzt den Fehler, ohne dass man die Lösung kennt. Ohne Konvexität gibt es Verfahren, aber keinen billigen Beweis.

Das Beispiel ist die **kleinste umschließende Kugel** einer Punktwolke in der Ebene. Ihr nichtkonvexes Gegenstück ist die **getrimmte Kugel**: die kleinste Kugel, die mindestens m der n Punkte enthält (Ausreißer dürfen draußen bleiben).

## Modell

- **Primal:** `min R` mit `‖xᵢ − c‖ ≤ R` für alle Punkte. Jeder Mittelpunkt c gibt eine **obere Schranke** (Abstand zum fernsten Punkt).
- **Dual:** `max_λ Σ λᵢ‖xᵢ‖² − ‖Σ λᵢxᵢ‖²` über Gewichte im Simplex (= R*²). Jedes Gewicht gibt eine **untere Schranke**: die Streuung der Punkte um ihren Gewichtsschwerpunkt (schwache Dualität, Beweis im 📐-Expander der App).
- **Verfahren:** Frank-Wolfe auf dem Dual (`kvz_algorithm.dual_run`): Der fernste Punkt bekommt mehr Gewicht, der Mittelpunkt rückt auf ihn zu, die Schrittweite folgt aus exakter Linienminimierung.
- **Orakel in der App:** der exakte Radius nach Welzl; ein unabhängiger Prüfer (`check_certificate`) läuft bei jedem Lauf und meldet eine verletzte Regel als Fehler.

## Methodik

Vier Experimente und eine Spielwiese, jedes mit live berechnetem Verdikt: (1) das Zertifikat im Verlauf, (2) die Zahl der Schritte bis zur Lücke ε über Punktzahl und Dimension, (3) lokal gleich global? (Zufallsstarts, konvex gegen getrimmt), (4) was der Beweis ohne Konvexität kostet, (🎲) schwache Dualität zum Anfassen (Mittelpunkt setzen, Gewichte würfeln).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Experimente laufen über die fünf festen Wolken `SWEEP_SEEDS` (je Wolkenart), nicht über einen einzelnen Seed. Die Standardwolke (Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Handinstanz (0,0), (4,0), (0,3), (1,1) | Kugel um (2; 1,5), Radius 2,5; dual mit Gewichten (0; ½; ½; 0): 6,25 = 2,5². Getrimmt (0,0), (1,0), (0,1), (10,0) mit m = 3: Optimum 0,7071, das Wechselverfahren aus (10, 0) bleibt bei 5 hängen |
| Standardwolke (20 Punkte, 200 Schritte) | untere Schranke 2,334, obere 2,346, wahr 2,340: bewiesene Lücke 0,51 %, wahrer Fehler 0,27 % |
| Schwache Dualität | In 1500 Zufallspaaren (Mittelpunkt, Gewichte) in 2, 10 und 50 Dimensionen keine Verletzung der Reihenfolge untere Schranke ≤ wahrer Radius ≤ obere Schranke |
| Wahrer Fehler gegen Lücke | Der Fehler der oberen Schranke ist im Median über den Verlauf **etwa die Hälfte der Lücke**: 0,500 bis 0,503 in 14 von 15 festen Wolken (drei Wolkenarten), 0,59 in einer Normalwolke (Seed 100001). Das Zertifikat überschätzt den Fehler also etwa um den Faktor 2. Beobachtung, kein Satz |
| Schritte bis zur relativen Lücke ε | **etwa 0,5 bis 0,9 mal 1/ε**, unabhängig von Punktzahl und Dimension (n·d von 100 bis 20 000): bei ε = 10⁻³ zwischen 682 und 866 Schritte; zehnmal mehr Genauigkeit kostet zehnmal mehr Schritte (Verhältnis 8 bis 13) |
| Konvexe lokale Suche (Mittelpunkt-Verfahren, 2000 Schritte) | Von allen 100 Zufallsstarts in jeder der 15 Wolken endet der Radius höchstens 0,1 % über dem wahren (schlechtester gemessen 0,05 %) |
| Getrimmte Kugel, Wechselverfahren, 100 Starts, n = 20, 4 aussortiert | Trefferquote des globalen Optimums im Median **10 % (Normalwolke), 29 % (Haufen + Ausreißer), 25 % (zwei Haufen)**, in einzelnen Wolken 0 bis 40 %; schlechtester Start im Median 35 % / 248 % / 10 % über dem Optimum, im Extremfall 277 %; 6 bis 9 verschiedene Endwerte im Median je Wolkenart. Die Zahlen gehören zu **diesem** Wechselverfahren und dieser Startverteilung |
| Aufwand des Beweises | Teilmengen, die der Beweis durch Aufzählen prüft: C(20, 4) = 4845, C(40, 6) = 3 838 380. Über Stützpunkte (Kugeln durch höchstens d + 1 Punkte): in der Ebene n = 20: 1330, n = 40: 10 660; in 10 Dimensionen bei n = 20: 784 605, bei n = 40 über 2 Milliarden |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Beweis ohne Konvexität ist nicht in jeder Dimension unbezahlbar.** Der Plan sagte „der einzige Beweis ist Aufzählen, und das wächst explosionsartig“. Das gilt für das Aufzählen der entfernten Punkte. In der **Ebene** gibt es einen schnellen exakten Weg (alle Kugeln durch zwei oder drei Punkte), der auf 20 Punkten dasselbe Optimum findet (einmal gemessen: 0,06 s gegen 4 bis 14 s für die Aufzählung, nicht getestet, weil zeit- und plattformabhängig). Die Zahl dieser Kugeln wächst aber mit n^(d+1). Die Demo zeigt beides, und die Aussage lautet jetzt: ohne Konvexität wächst der Beweis mit der Dimension; die Lücke des konvexen Falls kostet in jeder Dimension O(n·d) je Schritt.
- **SLSQP taugt nicht als Maßstab für „lokal gleich global“.** Phase 0 maß 0,1 bis 2,1 % numerische Ausfälle (Abweichungen bis 10³¹). Die App benutzt deshalb das Mittelpunkt-Verfahren; SLSQP bleibt als zweites Orakel im Test, mit Standardtoleranz (bei `ftol` = 10⁻¹⁴ scheiterte mehr als die Hälfte der Läufe).
- **„Fehler ≈ ½ Lücke“ gilt nicht in jeder Wolke.** Phase 0 sah über den Verlauf 0,37 bis 0,59; im fertigen Verdikt (Median über den Verlauf) liegt es in 14 von 15 festen Wolken bei 0,50 und in einer bei 0,59. Es gibt keinen Beweis dafür.
- **Zwei Fehler fanden die Randfalltests:** `trimmed_exact_support` gab bei m = 1 Mittelpunkt und Radius vertauscht zurück, und die Jahresangabe der Aufwandsschätzung schnitt Nachkommastellen ab. Beide sind behoben, Regressionstests stehen in `tests/test_edge_cases.py` und `tests/test_verdicts.py`.

## Ehrliche Grenzen

- Es ist **nur die euklidische Kugel in der Ebene** (für die Experimente zur Rate auch in höheren Dimensionen). Andere Formen sind andere Kegelprogramme (Stück 2).
- Die **Rate 0,5 bis 0,9 /ε** ist gemessen bis ε = 10⁻³. Für 10⁻⁶ wären es hochgerechnet rund 750 000 Schritte; das ist nicht gemessen. Sie passt zur bekannten Größenordnung O(1/ε) für Kernmengen der kleinsten Kugel (Bădoiu und Clarkson), ist aber hier nicht daraus abgeleitet.
- Das **Wechselverfahren** ist eine von vielen lokalen Suchen. Bessere Heuristiken (viele Starts, Verbesserungsschritte) treffen öfter, beweisen aber nichts.
- Die Laufzeit je Teilmenge im Experiment 4 ist auf dem jeweiligen Server gemessen und deshalb nicht reproduzierbar; nur die Zahlen der Teilmengen und Stützpunkt-Kugeln sind getestet.
- Die Hilfeaussage, dass die Slater-Bedingung hier erfüllt ist, ist eine Rechnung von Hand (Beweis im Expander), nicht Teil der Tests.

## Verifikation

- **Kern:** Schranken, ein Schritt und der zweite Schritt gegen Handrechnung auf der Mini-Instanz (`tests/conftest.py`, `tests/test_algorithm.py`); der Prüfer `check_certificate` fängt jede der vier verletzten Regeln.
- **Orakel:** `tests/test_oracle_konvexitaet_zertifikat.py`: Welzl gegen die Kreise durch zwei und drei Punkte (40 Wolken), gegen scipy-SLSQP (24 Wolken, 2D und 3D) und gegen Stützpunkt-Aufzählung in 3D; das Dual gegen scipy auf dem Simplex; Aufzählen gegen Stützpunkte für die getrimmte Kugel (30 Wolken in 2D, 6 in 3D).
- **Eigenschaften:** Skalierung, Verschiebung und Reihenfolge ändern nichts Wesentliches; schwache Dualität auf Zufallspaaren; beide Schranken umklammern den wahren Radius in jedem Schritt; die untere Schranke sinkt nie; die lokale Suche ist nie besser als das Optimum und nie schlechter als ihr Start.
- **Randfälle:** ein Punkt, genau zwei Punkte, identische Punkte, kollineare Punkte, null Schritte, ungültige Eingaben mit Meldung, inklusive Grenzen von m und r.
- **Verdikte:** reine Funktionen; Schwellen (1 % und 10 % Lücke, 50 % Trefferquote) und jede Zahl im Text in `tests/test_verdicts.py`.
- **Claims:** jede Zahl dieser README in `tests/test_claims.py`.
- **Presets, Oberfläche, Layout:** Presets vollständig und innerhalb der Grenzen, Permalink verwirft ungültige Werte, AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Schnellstart und fester Kopf, Footer, Laufzeit-Importe.
- **Umfang:** 100 Tests, rund 3,3 Minuten (2026-11-01, Windows; eine Hälfte davon sind AppTests und der Claims-Lauf über 15 feste Wolken).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs); geprüft ist gegen Orakel, Eigenschaften und Claims.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kvz_algorithm.py` | Kern: Schranken, Zertifikatsprüfer, Dualverfahren, Welzl, getrimmte Kugel (lokal, Aufzählung, Stützpunkte), Mittelpunkt-Verfahren |
| `kvz_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kvz_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kvz_layout.py` | Fester Kopf (Stylesheet) |
| `kvz_pdf_export.py` | PDF-Export der aktuellen Wolke (fpdf2) |
| `kvz_presets.py` | Permalink (Begrenzen, gültige Auswahlwerte), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kvz_scenario.py` | Punktwolken (Normalwolke, Haufen + Ausreißer, zwei Haufen) und Normalwolken in d Dimensionen |
| `kvz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Ellipsoid, Würfel und andere Formen**, **Kegel-Dualität mit Lücke** und **Innere-Punkte-Verfahren**: kommen in den Stücken 2, 4 bis 7 der Reihe.
- **Konvexitätsprüfung nach Regeln (DCP)** und **Ableitung durch Kegelprogramme**: optionale Stücke 3 und 15.
- **Ein allgemeiner Löser** (CVXPY, Clarabel): Die Reihe baut ihre Löser selbst; die Bibliotheken dienen nur als Gegenprobe in späteren Stücken.
- **Weitere Heuristiken für die getrimmte Kugel** (Mehrfachstart mit Verbesserung, Simulated Annealing): nicht das Thema.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
