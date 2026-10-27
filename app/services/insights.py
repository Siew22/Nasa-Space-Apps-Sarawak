"""Crew fire-safety takeaways for the dashboard's front page.

Nothing here is typed in: every number is recomputed from the processed NASA tables, the sweep formulas or the 2D
model, so the cards follow the data. Each item states its kind of evidence (real measurement or simulation), the
dataset it rests on, and the dashboard tab where the evidence can be inspected.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas import SimulationRequest
from app.services import flex_inference
from app.services.research_engine import (
    MIN_TESTS_TO_RANK,
    get_empirical_material_ranking,
    run_parametric_sensitivity_sweep,
)
from app.services.simulation_service import run_simulation

# FLEX oxygen groups (mole fraction) for tests without added CO2 or helium; NASA tested 12-34 % O2.
O2_LOW_MAX, O2_AIR = 0.19, (0.19, 0.23)
O2_RICH_MIN = 0.27
CO2_HIGH_MIN = 0.15  # "CO2-diluted": at least 15 % CO2
STANDARD_GRAVITY = 9.80665


def _flex_tests() -> pd.DataFrame:
    df = pd.read_csv(flex_inference.TESTS_PATH)
    df["k"] = pd.to_numeric(df["burning_rate_constant_mm2_s"], errors="coerce")
    df["end"] = df["test_end"].astype(str).str.strip()
    return df


def _went_out_share(df: pd.DataFrame) -> tuple[float, int]:
    """Share of non-disrupted tests whose flame went out while fuel remained, and how many tests that is."""
    done = df[df["end"].isin(["Extinction", "Completion"])]
    return float((done["end"] == "Extinction").mean()), int(len(done))


def crew_insights(lang: str = "en") -> list[dict[str, Any]]:
    """Six takeaways: key, icon, title, text (with the computed numbers), action, evidence, dataset, tab."""
    t = (lambda e, z: e) if lang == "en" else (lambda e, z: z)
    items: list[dict[str, Any]] = []

    # 1. Airflow (sweep formulas, no gravity, normal oxygen)
    pts = run_parametric_sensitivity_sweep("flow_speed", 0.0, 0.25, 25, fixed_gravity_m_s2=0.0,
                                           fixed_o2_pct=21.0, lang=lang)["curve_points"]
    burning = [p for p in pts if p["predicted_flame_intensity_proxy"] > 0]
    if burning and burning[0] is not pts[0]:
        x_on, x_max = burning[0]["sweep_value"], burning[-1]["sweep_value"]
        ratio = burning[-1]["predicted_flame_intensity_proxy"] / burning[0]["predicted_flame_intensity_proxy"]
        items.append({
            "key": "airflow", "icon": "🌬️", "evidence": "SIMULATED_RESULT", "dataset": None, "tab": "stem",
            "title": t("Airflow keeps a fire alive in orbit", "在轨道上，是气流让火持续燃烧"),
            "text": t(f"With no gravity there is no buoyancy to bring in oxygen. In the model, the flame is smothered below "
                      f"{x_on:.2f} m/s of airflow and spreads {ratio:.1f}× faster at {x_max:.2f} m/s.",
                      f"没有重力就没有浮力来输送氧气。在模型中，气流低于 {x_on:.2f} m/s 时火焰窒息熄灭，"
                      f"在 {x_max:.2f} m/s 时蔓延速度提高到 {ratio:.1f} 倍。"),
            "action": t("Switching the ventilation off starves a fire.", "关闭通风可以让火焰缺氧。"),
        })

    df = _flex_tests()
    plain = df[(df["x_co2"] == 0) & (df["x_he"] == 0) & df["k"].notna()]
    low = plain[plain["x_o2"] < O2_LOW_MAX]["k"]
    air = plain[(plain["x_o2"] >= O2_AIR[0]) & (plain["x_o2"] < O2_AIR[1])]["k"]
    rich = plain[plain["x_o2"] >= O2_RICH_MIN]["k"]

    # 2. Oxygen (real FLEX measurements)
    if len(low) and len(air) and len(rich):
        items.append({
            "key": "oxygen", "icon": "🫁", "evidence": "REAL_MEASUREMENT", "dataset": "FLEX", "tab": "ai",
            "title": t("More oxygen, faster fire", "氧气越多，燃烧越快"),
            "text": t(f"In NASA's FLEX tests without added CO₂ or helium, droplets burned at K = {low.mean():.2f} mm²/s below "
                      f"{O2_LOW_MAX:.0%} O₂ ({len(low)} tests), {air.mean():.2f} near normal air ({len(air)}) and "
                      f"{rich.mean():.2f} at {O2_RICH_MIN:.0%} O₂ or more ({len(rich)}): {rich.mean() / air.mean() - 1:+.0%} "
                      "versus normal air.",
                      f"在未添加 CO₂ 或氦气的 NASA FLEX 试验中，液滴燃烧速率 K 在 O₂ 低于 {O2_LOW_MAX:.0%} 时为 "
                      f"{low.mean():.2f} mm²/s（{len(low)} 次），接近正常空气时为 {air.mean():.2f}（{len(air)} 次），"
                      f"O₂ 达到 {O2_RICH_MIN:.0%} 及以上时为 {rich.mean():.2f}（{len(rich)} 次）：比正常空气 "
                      f"{rich.mean() / air.mean() - 1:+.0%}。"),
            "action": t("Oxygen-enriched cabins need stricter material and ventilation rules.",
                        "富氧舱室需要更严格的材料与通风规定。"),
        })

    # 3. CO2 dilution (real FLEX measurements)
    no_co2, co2 = df[df["x_co2"] == 0], df[df["x_co2"] >= CO2_HIGH_MIN]
    if len(no_co2) and len(co2):
        k0, k1 = no_co2["k"].mean(), co2["k"].mean()
        h0, _ = _went_out_share(no_co2[no_co2["is_heptane"] == 1])
        h1, _ = _went_out_share(co2[co2["is_heptane"] == 1])
        m0, _ = _went_out_share(no_co2[no_co2["is_heptane"] == 0])
        m1, _ = _went_out_share(co2[co2["is_heptane"] == 0])
        items.append({
            "key": "co2", "icon": "🧯", "evidence": "REAL_MEASUREMENT", "dataset": "FLEX", "tab": "ai",
            "title": t("CO₂ in the air slows the flame", "空气中的 CO₂ 会减缓火焰"),
            "text": t(f"FLEX tests with at least {CO2_HIGH_MIN:.0%} CO₂ burned at K = {k1:.2f} vs {k0:.2f} mm²/s without CO₂ "
                      f"({k1 / k0 - 1:+.0%}), at similar oxygen ({co2['x_o2'].mean():.0%} vs {no_co2['x_o2'].mean():.0%}). "
                      f"Heptane flames went out in {h1:.0%} of tests instead of {h0:.0%}; methanol flames went out almost "
                      f"every time either way ({m1:.0%} vs {m0:.0%}).",
                      f"CO₂ 至少为 {CO2_HIGH_MIN:.0%} 的 FLEX 试验燃烧速率 K 为 {k1:.2f}，无 CO₂ 时为 {k0:.2f} mm²/s"
                      f"（{k1 / k0 - 1:+.0%}），两者氧浓度相近（{co2['x_o2'].mean():.0%} 对 {no_co2['x_o2'].mean():.0%}）。"
                      f"正庚烷火焰熄灭的比例从 {h0:.0%} 升至 {h1:.0%}；甲醇火焰无论是否有 CO₂ 几乎都会熄灭"
                      f"（{m1:.0%} 对 {m0:.0%}）。"),
            "action": t("Diluting the air with an inert gas is a way to fight fire in a sealed cabin.",
                        "用惰性气体稀释空气是密闭舱内灭火的一种方法。"),
        })

    # 4. Self-extinction is not reliable (real FLEX measurements)
    droplets = get_empirical_material_ranking("liquid_droplet")
    ext = sum(r["outcome_counts"]["Extinction"] for r in droplets)
    comp = sum(r["outcome_counts"]["Completion"] for r in droplets)
    if ext + comp:
        items.append({
            "key": "self_extinction", "icon": "⚠️", "evidence": "REAL_MEASUREMENT", "dataset": "FLEX", "tab": "evidence",
            "title": t("Don't count on a fire putting itself out", "不要指望火会自己熄灭"),
            "text": t(f"{ext} of {ext + comp} FLEX droplet flames ({ext / (ext + comp):.0%}) went out on their own before the "
                      f"fuel was used up, but {comp} burned until the fuel was gone.",
                      f"{ext + comp} 次 FLEX 液滴火焰中有 {ext} 次（{ext / (ext + comp):.0%}）在燃料耗尽前自行熄灭，"
                      f"但仍有 {comp} 次一直燃烧到燃料耗尽。"),
            "action": t("Every fire needs an active response, even in microgravity.", "即使在微重力下，每一次起火都需要主动处置。"),
        })

    # 5. Toxic gas depends on how much can burn (real BASS-II measurements)
    solids = [r for r in get_empirical_material_ranking("solid_fuel")
              if r["empirical_test_count"] >= MIN_TESTS_TO_RANK and r["mean_final_co_ppm"] is not None
              and r["material_class"].startswith("pmma")]
    if len(solids) >= 2:
        hi, lo = solids[0], solids[-1]
        name = (lambda r: r["material_name"]) if lang == "en" else (lambda r: r["material_name_zh"])
        items.append({
            "key": "toxic_co", "icon": "☠️", "evidence": "REAL_MEASUREMENT", "dataset": "BASS_II", "tab": "evidence",
            "title": t("How much can burn matters for toxic gas", "可燃物的多少影响有毒气体的量"),
            "text": t(f"In BASS-II, {name(hi)} left {hi['mean_final_co_ppm']:.0f} ppm of carbon monoxide on average "
                      f"({hi['empirical_test_count']} tests) against {lo['mean_final_co_ppm']:.0f} ppm for {name(lo)} "
                      f"({lo['empirical_test_count']} tests): the same plastic, "
                      f"{hi['mean_final_co_ppm'] / lo['mean_final_co_ppm']:.0f}× more CO.",
                      f"在 BASS-II 中，{name(hi)} 平均留下 {hi['mean_final_co_ppm']:.0f} ppm 一氧化碳"
                      f"（{hi['empirical_test_count']} 次试验），{name(lo)} 为 {lo['mean_final_co_ppm']:.0f} ppm"
                      f"（{lo['empirical_test_count']} 次）：同一种塑料，CO 相差 {hi['mean_final_co_ppm'] / lo['mean_final_co_ppm']:.0f} 倍。"),
            "action": t("Limit how much burnable material sits in one place.", "限制同一位置存放的可燃材料数量。"),
        })

    # 6. No "up" in orbit (2D model, no airflow)
    def final_state(g: float) -> tuple[float, float, float]:
        frame = run_simulation(SimulationRequest(effective_gravity_m_s2=g, forced_flow_speed_m_s=0.0,
                                                 hot_gas_temperature_k=1200.0, duration_s=1.5, steps=15,
                                                 grid_resolution=15)).frames[-1]
        return frame.plume_centroid_m[1] * 100, frame.diffusion_radius_m * 100, frame.time_s

    rise_earth, _, dur = final_state(STANDARD_GRAVITY)
    rise_orbit, radius, _ = final_state(0.0)
    items.append({
        "key": "no_up", "icon": "🛰️", "evidence": "SIMULATED_RESULT", "dataset": None, "tab": "sandbox",
        "title": t("No “up” in orbit: heat and smoke don't rise", "轨道上没有“上”：热气和烟不会上升"),
        "text": t(f"In the 2D model, hot gas from a small fire rises {rise_earth:.0f} cm in {dur:.1f} s on Earth. With no "
                  f"gravity it rises {rise_orbit:.0f} cm and only spreads into a ball {2 * radius:.1f} cm wide.",
                  f"在 2D 模型中，小火产生的热气在地球上 {dur:.1f} 秒内上升 {rise_earth:.0f} cm；没有重力时上升 "
                  f"{rise_orbit:.0f} cm，只会扩散成直径 {2 * radius:.1f} cm 的球团。"),
        "action": t("Smoke goes where the airflow takes it, so detect and clear it along the air path, not at the ceiling.",
                    "烟会随气流流动，因此应沿气流路径而不是在天花板处探测和清除烟雾。"),
    })
    return items
