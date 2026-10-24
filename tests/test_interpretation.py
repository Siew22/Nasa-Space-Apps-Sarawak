"""Interpretation texts are built from the computed numbers, never typed in."""

import pytest

from app.services.research_engine import (
    MIN_TESTS_TO_RANK,
    get_empirical_material_ranking,
    ranking_insights,
    run_parametric_sensitivity_sweep,
)


def sweep(param, lo, hi, **fixed):
    return run_parametric_sensitivity_sweep(sweep_param=param, range_min=lo, range_max=hi, steps=25, lang="en", **fixed)


def test_airflow_sweep_text_quotes_its_own_smothering_threshold_and_growth():
    r = sweep("flow_speed", 0.0, 0.25, fixed_gravity_m_s2=0.0, fixed_o2_pct=21.0)
    pts = r["curve_points"]
    burning = [p for p in pts if p["predicted_flame_intensity_proxy"] > 0]
    text = " ".join(r["scientific_narration"])
    assert f"{burning[0]['sweep_value']:.2f} m/s" in text and "smothered" in text
    a, b = burning[0]["predicted_flame_intensity_proxy"], burning[-1]["predicted_flame_intensity_proxy"]
    assert f"{a:.3f} to {b:.3f}" in text and f"{b / a:.1f}×" in text
    assert "fans" in r["safety_note"]


def test_no_smothering_claim_when_nothing_is_smothered():
    r = sweep("flow_speed", 0.0, 0.25, fixed_gravity_m_s2=9.81, fixed_o2_pct=21.0)
    assert all(p["predicted_flame_intensity_proxy"] > 0 for p in r["curve_points"])
    assert "smothered" not in " ".join(r["scientific_narration"])


def test_text_changes_with_the_inputs():
    a = sweep("gravity", 0.0, 10.0, fixed_flow_speed_m_s=0.05, fixed_o2_pct=21.0)["scientific_narration"]
    b = sweep("gravity", 0.0, 10.0, fixed_flow_speed_m_s=0.30, fixed_o2_pct=21.0)["scientific_narration"]
    assert a != b


@pytest.mark.parametrize("u", [0.01, 0.05, 0.3])
def test_buoyancy_crossover_is_where_ri_equals_one(u):
    # Model: Ri = g * beta * dT * L / u^2 with beta = 1/300 K^-1, dT = 1200 K, L = 0.02 m (research_engine.py)
    g_star = u * u / ((1 / 300) * 1200 * 0.02)
    text = " ".join(sweep("gravity", 0.0, 10.0, fixed_flow_speed_m_s=u, fixed_o2_pct=21.0)["scientific_narration"])
    assert f"g ≈ {g_star:.2g} m/s²" in text


def test_bass_ii_insights_match_the_ranking_rows():
    rows = get_empirical_material_ranking("solid_fuel")
    ins = ranking_insights("solid_fuel", rows, "en")
    ranked = [r for r in rows if r["empirical_test_count"] >= MIN_TESTS_TO_RANK]
    top, bot = ranked[0], ranked[-1]
    assert f"{top['material_name']}, mean **{top['mean_final_co_ppm']:.0f} ppm**" in ins["points"][0]
    assert f"{bot['material_name']}, **{bot['mean_final_co_ppm']:.0f} ppm**" in ins["points"][0]
    few = [r for r in rows if r["empirical_test_count"] < MIN_TESTS_TO_RANK]
    assert all(f"{r['material_name']} ({r['empirical_test_count']})" in ins["points"][-1] for r in few)
    n_over = sum(r["co_over_range_readings"] for r in rows)
    assert any(f"{n_over} CO readings" in p for p in ins["points"])


def test_flex_insights_match_the_outcome_counts():
    rows = get_empirical_material_ranking("liquid_droplet")
    ins = ranking_insights("liquid_droplet", rows, "en")
    ext = sum(r["outcome_counts"]["Extinction"] for r in rows)
    comp = sum(r["outcome_counts"]["Completion"] for r in rows)
    assert f"({ext} of {ext + comp})" in ins["safety"] and f"**{comp} burned" in ins["safety"]
    for r in rows:
        c = r["outcome_counts"]
        assert f"({c['Extinction']}/{c['Extinction'] + c['Completion']})" in ins["points"][1]


def test_insights_exist_in_both_languages_and_handle_empty_input():
    rows = get_empirical_material_ranking("solid_fuel")
    en, zh = ranking_insights("solid_fuel", rows, "en"), ranking_insights("solid_fuel", rows, "zh")
    assert len(en["points"]) == len(zh["points"]) and en["points"] != zh["points"]
    assert ranking_insights("solid_fuel", [], "en") == {"points": [], "safety": "", "caveat": ""}
