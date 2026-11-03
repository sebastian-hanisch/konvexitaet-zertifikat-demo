r"""Texte dürfen keine Steuerzeichen enthalten (ein v-Escape im r-String wurde einmal zum Tabulator 0x0B und zerstörte \varepsilon)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ERLAUBT = {"\t", "\n", "\r"}
DATEIEN = sorted([ROOT / "README.md", *ROOT.glob("*.py")], key=lambda p: p.name)


@pytest.mark.parametrize("path", DATEIEN, ids=lambda p: p.name)
def test_text_enthaelt_kein_steuerzeichen(path):
    text = path.read_text(encoding="utf-8")
    bad = sorted({(hex(ord(ch)), text[:i].count("\n") + 1) for i, ch in enumerate(text) if ord(ch) < 32 and ch not in ERLAUBT})
    assert not bad, f"{path.name}: Steuerzeichen (Code, Zeile) {bad}"


def test_literaturangabe_hat_varepsilon():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "$O(1/\\varepsilon)$" in text
