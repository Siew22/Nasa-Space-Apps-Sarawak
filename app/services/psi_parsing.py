"""Documented rules for reading the official NASA PSI tables (BASS-II PSI-25, FLEX PSI-69).

The raw files are stored unchanged; these rules only INTERPRET cells, and every non-numeric
cell is classified rather than silently dropped or guessed.

Sources for the rules:
  * BASS-II_ReadMe.docx (PSI-25): "OR = over range (> 1000 ppm)" for the CO sensor.
  * NASA/TP-2015-216046, Appendix A.1 (FLEX): "-" means no value, e.g. no extinction diameter
    because the droplet burned to completion or disrupted.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Cell flags returned by numeric_cell()
OK, APPROXIMATE, MISSING, NO_VALUE, OVER_RANGE, CENSORED_ABOVE, MULTIPLE_VALUES, TEXT = (
    "ok", "approximate", "missing", "no_value", "over_range", "censored_above", "multiple_values", "text")

_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def numeric_cell(value: Any) -> tuple[float | None, str]:
    """Interpret one table cell as a number, returning (value or None, flag)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, MISSING
    if isinstance(value, (int, float)):
        return float(value), OK
    text = str(value).strip()
    low = text.lower()
    if text == "" or low == "none":
        return None, MISSING
    if text in {"–", "-", "—"}:
        return None, NO_VALUE
    if low in {"or", "over"}:
        return None, OVER_RANGE
    if text.startswith(">") and _NUMBER.match(text[1:].strip()):
        return None, CENSORED_ABOVE
    if text.startswith("~") and _NUMBER.match(text[1:].strip()):
        return float(text[1:].strip()), APPROXIMATE
    if _NUMBER.match(text):
        return float(text), OK
    if len(re.findall(r"\d+(?:\.\d+)?", text)) > 1:
        return None, MULTIPLE_VALUES
    return None, TEXT


MATERIAL_CLASSES = {
    "pmma_rod_clear": ("Clear PMMA rod", "透明 PMMA 棒"),
    "pmma_rod_black": ("Black PMMA rod", "黑色 PMMA 棒"),
    "pmma_thin_film": ("Thin PMMA film (0.1–0.4 mm)", "PMMA 薄膜（0.1–0.4 毫米）"),
    "pmma_slab": ("PMMA slab (1–5 mm)", "PMMA 板（1–5 毫米）"),
    "sibal_fabric": ("SIBAL cotton-fiberglass fabric", "SIBAL 棉-玻璃纤维织物"),
    "nomex_fabric": ("Nomex III fabric", "Nomex III 织物"),
    "pmma_sphere_section": ("PMMA spherical section (4 cm)", "PMMA 球面截段（4 厘米）"),
    "wax_candle": ("Japan wax candle", "日本蜡烛"),
    "other": ("Other material", "其他材料"),
}


def bass_material_class(text: Any) -> str:
    """Map a free-text 'Fuel Sample Material' cell to one class key (ordered, documented rules)."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if "candle" in t:
        return "wax_candle"
    if "spher" in t:
        return "pmma_sphere_section"
    if "nomex" in t:
        return "nomex_fabric"
    if "sibal" in t:
        return "sibal_fabric"
    if "rod" in t:
        return "pmma_rod_black" if "bla" in t else "pmma_rod_clear"
    if "micron" in t or "film" in t or re.search(r"\b0\.\d+ ?mm\b", t):
        return "pmma_thin_film"
    if "slab" in t or re.search(r"\b\d+(\.\d+)? ?mm thick\b", t) or re.search(r"\b\d ?mm\b", t):
        return "pmma_slab"
    return "other"
