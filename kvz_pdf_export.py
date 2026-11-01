"""PDF-Export der aktuellen Wolke mit beiden Schranken (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kvz_evaluation import Analysis, de, pct

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "λ": "lambda", "²": "2"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kleinste umschließende Kugel mit Zertifikat"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Wolke: {s.cloud}, {s.n} Punkte in der Ebene, {s.steps} Schritte des Dualverfahrens, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Obere Schranke (Kugel um den Mittelpunkt c = ({de(a.run.centre[0], 3)}; {de(a.run.centre[1], 3)})): {de(a.upper, 4)}.",
        f"Untere Schranke (dualer Wert der Gewichte): {de(a.lower, 4)}.",
        f"Bewiesene Lücke: {pct(a.gap_rel)} der oberen Schranke.",
        f"Wahrer kleinster Radius (exakte Rechnung nach Welzl): {de(a.exact_r, 4)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "x", "y", "Gewicht im Dual"], [18, 36, 36, 44]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.inst.points[i, 0], 3), de(a.inst.points[i, 1], 3), de(a.run.lam[i], 4)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Jedes Gewicht im Simplex gibt eine untere, jeder Mittelpunkt eine obere Schranke für den kleinsten Radius (schwache Dualität). "
                            "Quelle: konvexitaet-zertifikat-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
