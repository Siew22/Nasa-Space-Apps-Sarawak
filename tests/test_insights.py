"""Front-page crew takeaways: every number is recomputed here from the data or the models and must match the text."""

import pandas as pd
import pytest

from app.services import flex_inference
from app.services.insights import CO2_HIGH_MIN, O2_AIR, O2_LOW_MAX, O2_RICH_MIN, crew_insights
from app.services.research_engine import get_empirical_material_ranking

ITEMS = {it["key"]: it for it in crew_insights("en")}
FLEX = pd.read_csv(flex_inference.TESTS_PATH)
FLEX["k"] = pd.to_numeric(FLEX["burning_rate_constant_mm2_s"], errors="coerce")


def test_six_takeaways_each_with_evidence_type_and_tab():
    assert list(ITEMS) == ["airflow", "oxygen", "co2", "self_extinction", "toxic_co", "no_up"]
    for it in ITEMS.values():
        assert it["evidence"] in {"REAL_MEASUREMENT", "SIMULATED_RESULT"}
        assert (it["dataset"] is None) == (it["evidence"] == "SIMULATED_RESULT")
        assert it["tab"] in {"evidence", "stem", "sandbox", "audit", "ai"}
        assert it["title"] and it["text"] and it["action"]


def test_oxygen_numbers_are_the_flex_group_means():
    plain = FLEX[(FLEX.x_co2 == 0) & (FLEX.x_he == 0) & FLEX.k.notna()]
    low = plain[plain.x_o2 < O2_LOW_MAX].k
    air = plain[(plain.x_o2 >= O2_AIR[0]) & (plain.x_o2 < O2_AIR[1])].k
    rich = plain[plain.x_o2 >= O2_RICH_MIN].k
    text = ITEMS["oxygen"]["text"]
    assert f"K = {low.mean():.2f} mm²/s" in text and f"({len(low)} tests)" in text
    assert f"{air.mean():.2f} near normal air ({len(air)})" in text and f"{rich.mean():.2f} at" in text
    assert rich.mean() > air.mean() > low.mean()  # the claim "more oxygen, faster fire" holds in the data


def test_co2_numbers_and_the_methanol_caveat():
    no, yes = FLEX[FLEX.x_co2 == 0], FLEX[FLEX.x_co2 >= CO2_HIGH_MIN]
    text = ITEMS["co2"]["text"]
    assert f"K = {yes.k.mean():.2f} vs {no.k.mean():.2f} mm²/s" in text
    assert yes.k.mean() < no.k.mean()  # "slows the flame"
    assert "methanol" in text


def test_self_extinction_counts_match_the_ranking():
    rows = get_empirical_material_ranking("liquid_droplet")
    ext = sum(r["outcome_counts"]["Extinction"] for r in rows)
    comp = sum(r["outcome_counts"]["Completion"] for r in rows)
    assert ITEMS["self_extinction"]["text"].startswith(f"{ext} of {ext + comp} FLEX droplet flames")
    assert f"but {comp} burned" in ITEMS["self_extinction"]["text"]


def test_toxic_co_numbers_match_the_ranking():
    pmma = [r for r in get_empirical_material_ranking("solid_fuel")
            if r["empirical_test_count"] >= 5 and r["material_class"].startswith("pmma")]
    hi, lo = pmma[0], pmma[-1]
    text = ITEMS["toxic_co"]["text"]
    assert f"{hi['mean_final_co_ppm']:.0f} ppm" in text and f"{lo['mean_final_co_ppm']:.0f} ppm" in text


def test_no_up_rests_on_the_model_with_and_without_gravity():
    text = ITEMS["no_up"]["text"]
    assert "rises 0 cm" in text and "on Earth" in text


@pytest.mark.parametrize("key", ["airflow", "oxygen", "co2", "self_extinction", "toxic_co", "no_up"])
def test_chinese_version_has_the_same_cards(key):
    zh = {it["key"]: it for it in crew_insights("zh")}
    assert zh[key]["evidence"] == ITEMS[key]["evidence"] and zh[key]["tab"] == ITEMS[key]["tab"]
    assert zh[key]["text"] != ITEMS[key]["text"]
