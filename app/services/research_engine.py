from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from app.services.psi_parsing import MATERIAL_CLASSES, bass_material_class, numeric_cell

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT / "data" / "processed" / "dashboard_records.csv"


def _num(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Numeric values (NaN where not a clean number) and the flag for every cell (see psi_parsing)."""
    parsed = series.map(numeric_cell)
    return parsed.map(lambda p: p[0]).astype(float), parsed.map(lambda p: p[1])


def _round(x: float | None, digits: int) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), digits)


def get_empirical_material_ranking(track_key: str = "solid_fuel") -> list[dict[str, Any]]:
    """Per-material (BASS-II) or per-fuel (FLEX) aggregates computed from data/processed/dashboard_records.csv.

    Every number is computed from the loaded table; nothing is typed in. Whether that table is real is
    decided by src/provenance_audit.py (reports/data_provenance_audit.json), not here. Non-numeric cells
    are classified with app/services/psi_parsing.py and counted, never silently turned into numbers.
    """
    if not PROCESSED_CSV.exists():
        return []
    df = pd.read_csv(PROCESSED_CSV, low_memory=False)

    if track_key == "solid_fuel":
        sub = df[df["dataset_key"] == "BASS_II"].copy()
        if sub.empty:
            return []
        sub["material_class"] = sub["fuel_sample_material"].map(bass_material_class)
        sub["co_ppm"], sub["co_flag"] = _num(sub["final_co_ppm"])
        o2_in, _ = _num(sub["calibrated_initial_o2_pct_by_vol"])
        o2_out, _ = _num(sub["calibrated_final_o2_pct_by_vol"])
        sub["o2_drop"] = o2_in - o2_out
        rows = []
        for cls, g in sub.groupby("material_class"):
            label_en, label_zh = MATERIAL_CLASSES.get(cls, MATERIAL_CLASSES["other"])
            rows.append({
                "material_class": cls,
                "material_name": label_en,
                "material_name_zh": label_zh,
                "raw_material_spellings": sorted({" ".join(str(s).split()) for s in g["fuel_sample_material"]}),
                "empirical_test_count": int(len(g)),
                "co_numeric_readings": int((g["co_flag"] == "ok").sum()),
                "co_over_range_readings": int(g["co_flag"].isin(["over_range", "censored_above"]).sum()),
                "mean_final_co_ppm": _round(g["co_ppm"].mean(), 2),
                "max_final_co_ppm": _round(g["co_ppm"].max(), 1),
                "mean_o2_depletion_pct_vol": _round(g["o2_drop"].mean(), 3),
                "data_source": "; ".join(sorted(g["source_file"].dropna().astype(str).unique())),
            })
        rows.sort(key=lambda r: (r["mean_final_co_ppm"] is None, -(r["mean_final_co_ppm"] or 0)))
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        return rows

    sub = df[df["dataset_key"] == "FLEX"].copy()
    if sub.empty:
        return []
    sub["k"], _ = _num(sub["burning_rate_mm"])                         # mm^2/s (NASA/TP-2015-216046 Table IX)
    sub["d0"], _ = _num(sub["droplet_initial_diameter_mm"])
    sub["d_ext"], _ = _num(sub["visible_flame_extinction_diameter_mm"])
    outcome = sub["test_end"].astype(str).str.strip()
    rows = []
    for fuel, g in sub.groupby("fuel"):
        o = outcome.loc[g.index]
        ext, comp, disr = int((o == "Extinction").sum()), int((o == "Completion").sum()), int((o == "Disruption").sum())
        rows.append({
            "fuel_name": str(fuel),
            "empirical_test_count": int(len(g)),
            "mean_burning_rate_constant_mm2_s": _round(g["k"].mean(), 3),
            "burning_rate_constant_readings": int(g["k"].notna().sum()),
            "mean_initial_diameter_mm": _round(g["d0"].mean(), 2),
            # Extinction diameter is only meaningful when the flame actually went out.
            "mean_extinction_diameter_mm": _round(g.loc[o == "Extinction", "d_ext"].mean(), 2),
            "outcome_counts": {"Extinction": ext, "Completion": comp, "Disruption": disr},
            # Disrupted tests ended for another reason, so they are excluded rather than counted as "no extinction".
            "extinction_share_of_non_disrupted": _round(ext / (ext + comp), 3) if ext + comp else None,
            "data_source": "; ".join(sorted(g["source_file"].dropna().astype(str).unique())),
        })
    rows.sort(key=lambda r: (r["mean_burning_rate_constant_mm2_s"] is None, -(r["mean_burning_rate_constant_mm2_s"] or 0)))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


# Every figure in the texts below is read from the computed points or the ranking rows; nothing is typed in.
G_MOON, G_MARS, G_EARTH = 1.62, 3.71, 9.81  # m/s^2


def _sweep_narration(sweep_param: str, points: list[dict], g: float, u: float, o2: float, ri_coef: float,
                     en: bool) -> tuple[list[str], str]:
    """What this sweep shows, as bullet sentences built from its own numbers, plus one crew-safety takeaway.

    ri_coef is the model's Ri = ri_coef * g / u^2, so the Ri = 1 crossover is solved exactly (log Ri in the points is
    rounded and clipped at ±4).
    """
    t = (lambda e, z: e) if en else (lambda e, z: z)
    xs = [p["sweep_value"] for p in points]
    idx = [p["predicted_flame_intensity_proxy"] for p in points]
    burning = [i for i, v in enumerate(idx) if v > 0]
    smothered = [i for i, v in enumerate(idx) if v == 0]
    bx, by = [xs[i] for i in burning], [idx[i] for i in burning]
    lines: list[str] = []

    def at(x: float) -> float:
        return float(np.interp(x, bx, by))

    if sweep_param == "flow_speed":
        if smothered and burning:
            x_on = xs[burning[0]]
            lines.append(t(f"Below **{x_on:.2f} m/s** of airflow the flame is **smothered** (spread index 0)"
                           + (": with no gravity there is no buoyancy, so only the airflow can bring fresh oxygen."
                              if g == 0 else "."),
                           f"气流低于 **{x_on:.2f} m/s** 时火焰**窒息熄灭**（蔓延指数为 0）"
                           + ("：没有重力就没有浮力，只有气流能带来新鲜氧气。" if g == 0 else "。")))
        elif smothered:
            lines.append(t("The flame is smothered at every airflow speed in this range.", "在此范围内的所有风速下火焰都会窒息熄灭。"))
        if len(burning) >= 2:
            a, b = by[0], by[-1]
            lines.append(t(f"From {bx[0]:.2f} to {bx[-1]:.2f} m/s the spread index rises from **{a:.3f} to {b:.3f}** "
                           f"(**{b / a:.1f}×**): more airflow brings more oxygen to the flame.",
                           f"风速从 {bx[0]:.2f} 增至 {bx[-1]:.2f} m/s，蔓延指数由 **{a:.3f} 升至 {b:.3f}**（**{b / a:.1f} 倍**）："
                           "气流越强，送到火焰的氧气越多。"))
        u_star = None if g == 0 else float(np.sqrt(ri_coef * g))
        if g == 0:
            lines.append(t("With g = 0 there is no buoyancy at all, so the flow is **airflow-driven at every speed**.",
                           "g = 0 时完全没有浮力，因此在任何风速下流动都**由气流主导**。"))
        elif u_star is not None and xs[0] < u_star < xs[-1]:
            lines.append(t(f"Below about **{u_star:.3f} m/s** buoyancy is stronger than the airflow (Ri > 1); "
                           "above it the airflow takes over.",
                           f"风速低于约 **{u_star:.3f} m/s** 时浮力强于气流（Ri > 1）；高于此值则由气流主导。"))
        elif u_star is not None and u_star >= xs[-1]:
            lines.append(t(f"At g = {g:.2f} m/s² buoyancy beats the airflow at every speed up to {xs[-1]:.2f} m/s "
                           f"(Ri = 1 would need about {u_star:.2f} m/s).",
                           f"在 g = {g:.2f} m/s² 时，直到 {xs[-1]:.2f} m/s 浮力都强于气流（要达到 Ri = 1 约需 {u_star:.2f} m/s）。"))
        if g == 0 and smothered:
            safety = t("**For crew safety:** in orbit, airflow is what keeps a fire alive. Switching the fans off cuts its "
                       "oxygen supply, which is why spacecraft fire procedures stop the ventilation.",
                       "**对乘员安全的意义：** 在轨道上，是气流让火焰持续燃烧。关闭风扇就切断了它的氧气供应，"
                       "这正是航天器火灾处置程序要求停止通风的原因。")
        elif g == 0:
            safety = t(f"**For crew safety:** at {o2:.0f}% O₂ this model keeps the flame alive even at very low airflow, "
                       "so switching the fans off is not enough on its own in an oxygen-rich cabin.",
                       f"**对乘员安全的意义：** 在 {o2:.0f}% 氧浓度下，该模型中即使气流很弱火焰也能维持，"
                       "因此在富氧舱内仅关闭风扇并不足够。")
        else:
            safety = t("**For crew safety:** with gravity, buoyancy feeds the flame even with the fans off, so stopping "
                       "the airflow does not starve it the way it does in orbit.",
                       "**对乘员安全的意义：** 有重力时，即使关闭风扇，浮力仍会为火焰供氧，因此停止通风无法像在轨道上那样让火焰缺氧。")
        return lines, safety

    if sweep_param == "gravity":
        if smothered and burning and xs[smothered[0]] == 0:
            lines.append(t(f"With **no gravity** and only {u:.2f} m/s of airflow the flame is **smothered**. With any "
                           "gravity at all, buoyancy starts bringing in oxygen and it burns.",
                           f"**没有重力**且气流只有 {u:.2f} m/s 时火焰**窒息熄灭**；只要有一点重力，浮力就开始输送氧气，火焰便能燃烧。"))
        if len(burning) >= 2:
            places = [(t("in orbit", "轨道上"), idx[0])] + [
                (name, at(gx)) for name, gx in ((t("Moon", "月球"), G_MOON), (t("Mars", "火星"), G_MARS),
                                                (t("Earth", "地球"), G_EARTH)) if bx[0] <= gx <= bx[-1]]
            lines.append(t("Spread index: " + " · ".join(f"{n} **{v:.3f}**" for n, v in places) + ".",
                           "蔓延指数：" + " · ".join(f"{n} **{v:.3f}**" for n, v in places) + "。"))
        g_star = max(0.001, u) ** 2 / ri_coef  # same airflow floor as the sweep
        if g_star is not None:
            lines.append(t(f"Buoyancy overtakes the {u:.2f} m/s airflow (Ri = 1) at only **g ≈ {g_star:.2g} m/s²**, "
                           f"about {100 * g_star / G_EARTH:.2g}% of Earth gravity"
                           + (", so on the Moon and Mars flames are buoyancy-driven, much like on Earth."
                              if g_star < G_MOON else "."),
                           f"浮力在仅 **g ≈ {g_star:.2g} m/s²**（约为地球重力的 {100 * g_star / G_EARTH:.2g}%）时就超过 {u:.2f} m/s 的气流（Ri = 1）"
                           + ("，因此在月球和火星上火焰也由浮力主导，与地球相近。" if g_star < G_MOON else "。")))
        safety = t("**For crew safety:** a material tested on Earth burns in a buoyancy-driven flow, but in orbit the same "
                   "material burns in an airflow-driven one, so Earth tests alone cannot show how a fire behaves in space.",
                   "**对乘员安全的意义：** 材料在地球上测试时处于浮力主导的流动中，而在轨道上同一材料处于气流主导的流动中，"
                   "因此仅靠地面试验无法说明火焰在太空中的行为。")
        return lines, safety

    # oxygen
    if smothered and burning:
        lines.append(t(f"With no gravity and only {u:.2f} m/s of airflow the flame is **smothered at {xs[smothered[-1]]:.1f}% "
                       f"O₂ and below**, and burns from {xs[burning[0]]:.1f}%. (This is a hard cut-off built into the simple "
                       "model, not a measured limit.)",
                       f"没有重力且气流只有 {u:.2f} m/s 时，火焰在 **{xs[smothered[-1]]:.1f}% 及以下的氧浓度时窒息熄灭**，"
                       f"从 {xs[burning[0]]:.1f}% 起开始燃烧。（这是简化模型内置的硬阈值，并非实测极限。）"))
    if len(burning) >= 2:
        lo = 21.0 if bx[0] <= 21.0 else bx[0]
        hi = min(34.0, bx[-1])
        a, b = at(lo), at(hi)
        lines.append(t(f"Raising O₂ from {lo:.0f}%" + (" (normal air)" if lo == 21.0 else "") + f" to {hi:.0f}% raises "
                       f"the spread index from **{a:.3f} to {b:.3f} (+{b / a - 1:.0%})**. This model scales in a straight "
                       "line with oxygen; it does not include real flammability limits.",
                       f"氧浓度从 {lo:.0f}%" + ("（正常空气）" if lo == 21.0 else "") + f" 提高到 {hi:.0f}%，蔓延指数由 "
                       f"**{a:.3f} 升至 {b:.3f}（+{b / a - 1:.0%}）**。该模型与氧浓度呈线性关系，并未包含真实的可燃极限。"))
    regime = points[-1]["flow_regime"]
    lines.append(t(f"Oxygen does not change the flow type: at g = {g:.2f} m/s² and {u:.2f} m/s the burning flame stays "
                   f"**{regime}**.",
                   f"氧浓度不改变流动类型：在 g = {g:.2f} m/s²、风速 {u:.2f} m/s 时，燃烧中的火焰始终为 **{regime}**。"))
    safety = t("**For crew safety:** more oxygen means a stronger fire at the same airflow, so oxygen-enriched cabins "
               "(for example around 34% O₂, considered for reduced-pressure exploration cabins) need stricter material "
               "and ventilation rules.",
               "**对乘员安全的意义：** 在相同气流下，氧气越多火势越强，因此富氧舱室（例如低压探索舱考虑采用的约 34% 氧浓度）"
               "需要更严格的材料与通风规定。")
    return lines, safety


def run_parametric_sensitivity_sweep(
    sweep_param: str,
    range_min: float,
    range_max: float,
    steps: int,
    fixed_gravity_m_s2: float = 0.0,
    fixed_flow_speed_m_s: float = 0.05,
    fixed_o2_pct: float = 21.0,
    fixed_pressure_kpa: float = 101.3,
    lang: str = "zh"
) -> dict[str, Any]:
    """
    面向 STEM 教学的参数敏感性动态扫描与第一性原理解析：
    计算流动与燃烧无量纲数随变量的连续响应曲线，并输出物理机制叙述。
    """
    en = lang == "en"
    sweep_values = np.linspace(range_min, range_max, steps)
    curve_points = []
    
    char_l = 0.02       # 2 cm 特征尺寸
    air_viscosity = 1.5e-5
    delta_t = 1200.0
    beta = 1.0 / 300.0

    for val in sweep_values:
        # 确定当前点参数
        g = val if sweep_param == "gravity" else fixed_gravity_m_s2
        u = val if sweep_param == "flow_speed" else fixed_flow_speed_m_s
        o2 = val if sweep_param == "oxygen" else fixed_o2_pct
        p = fixed_pressure_kpa

        u_safe = max(0.001, u)

        # 流体力学无量纲量计算
        reynolds = (u_safe * char_l) / air_viscosity
        # 理查森数 Ri = Gr / Re^2
        ri = (g * beta * delta_t * char_l) / (u_safe ** 2) if u_safe > 0 else 1e4
        log_ri = float(np.round(np.log10(np.clip(ri, 1e-4, 1e4)), 2))

        # 浮力速度尺度
        u_buoy = float(np.round(np.sqrt(max(0.0, 2.0 * g * char_l * (delta_t / 1500.0))), 4))

        # 有效供氧通量 (Oxygen Flux Index ~ X_O2 * (u_forced + u_buoy))
        total_effective_flow = u + (0.5 * u_buoy)
        oxygen_flux = (o2 / 100.0) * total_effective_flow * 100.0

        # 估算特征火焰蔓延强度代理指标 (Spread Index ~ Oxygen Flux / (Heat Loss Factor))
        # 当 u 极低且 g=0 时，扩散受限发生窒息，蔓延强度骤降至 0
        if g == 0.0 and u < 0.02 and o2 <= 21.0:
            flame_spread_index = 0.0
            regime = "Quenching / Starvation" if en else "窒息自熄区 (Quenching / Starvation)"
        else:
            flame_spread_index = round(0.1 + (oxygen_flux * 0.035), 4)
            if log_ri < 0.0:
                regime = "Forced flow dominated" if en else "强迫对流主导 (Forced Flow)"
            else:
                regime = "Buoyant plume dominated" if en else "浮力自然对流主导 (Buoyant Plume)"

        curve_points.append({
            "sweep_value": round(float(val), 4),
            "reynolds_number": round(float(reynolds), 1),
            "log_richardson_number": log_ri,
            "buoyancy_velocity_scale_m_s": u_buoy,
            "oxygen_flux_index": round(float(oxygen_flux), 3),
            "predicted_flame_intensity_proxy": flame_spread_index,
            "flow_regime": regime
        })

    # Narration for STEM students, written from this sweep's own numbers
    narration, safety_note = _sweep_narration(sweep_param, curve_points, fixed_gravity_m_s2, fixed_flow_speed_m_s,
                                              fixed_o2_pct, beta * delta_t * char_l, en)

    return {
        "sweep_parameter": sweep_param,
        "curve_points": curve_points,
        "scientific_narration": narration,
        "safety_note": safety_note,
    }


MIN_TESTS_TO_RANK = 5


def ranking_insights(track_key: str, rows: list[dict[str, Any]], lang: str = "en") -> dict[str, Any]:
    """What the empirical ranking means, as sentences computed from its rows (see get_empirical_material_ranking)."""
    en = lang == "en"
    t = (lambda e, z: e) if en else (lambda e, z: z)
    points: list[str] = []
    if not rows:
        return {"points": points, "safety": "", "caveat": ""}

    if track_key == "solid_fuel":
        def name(r: dict) -> str:
            return r["material_name"] if en else r["material_name_zh"]

        ranked = [r for r in rows if r["empirical_test_count"] >= MIN_TESTS_TO_RANK and r["mean_final_co_ppm"] is not None]
        few = [r for r in rows if r["empirical_test_count"] < MIN_TESTS_TO_RANK]
        safety = ""
        if len(ranked) >= 2:
            top, bot = ranked[0], ranked[-1]
            ratio = top["mean_final_co_ppm"] / bot["mean_final_co_ppm"]
            points.append(t(f"**Most CO:** {name(top)}, mean **{top['mean_final_co_ppm']:.0f} ppm** over "
                            f"{top['empirical_test_count']} tests. **Least:** {name(bot)}, **{bot['mean_final_co_ppm']:.0f} ppm** "
                            f"over {bot['empirical_test_count']} tests (**{ratio:.1f}×** less).",
                            f"**CO 最多：** {name(top)}，{top['empirical_test_count']} 次试验平均 **{top['mean_final_co_ppm']:.0f} ppm**。"
                            f"**最少：** {name(bot)}，{bot['empirical_test_count']} 次试验平均 **{bot['mean_final_co_ppm']:.0f} ppm**"
                            f"（少 **{ratio:.1f} 倍**）。"))
            pmma = [r for r in ranked if r["material_class"].startswith("pmma")]
            if len(pmma) >= 2:
                hi, lo = pmma[0], pmma[-1]
                both_ends = top is hi and bot is lo
                points.append(t(f"**The same plastic, PMMA, spans {lo['mean_final_co_ppm']:.0f}–{hi['mean_final_co_ppm']:.0f} ppm** "
                                f"({name(lo)} vs {name(hi)})"
                                + (": the lowest and the highest in the whole ranking. In these tests the form and amount "
                                   "of material mattered more than which material it was." if both_ends else "."),
                                f"**同一种塑料 PMMA 的 CO 从 {lo['mean_final_co_ppm']:.0f} 到 {hi['mean_final_co_ppm']:.0f} ppm 不等**"
                                f"（{name(lo)} 对比 {name(hi)}）"
                                + ("：恰好是整个排名的最低与最高。在这些试验中，材料的形态与用量比材料种类更重要。" if both_ends else "。")))
                times = hi["mean_final_co_ppm"] / lo["mean_final_co_ppm"]
                safety = t(f"**For crew safety:** {name(hi)} left **{times:.0f}× more CO** than {name(lo)} of the same "
                           "plastic, so toxic-gas risk depends on how much material can burn, not only on what it is.",
                           f"**对乘员安全的意义：** 同为 PMMA，{name(hi)} 留下的 CO 是 {name(lo)} 的 **{times:.0f} 倍**，"
                           "因此有毒气体风险取决于可燃材料的多少，而不仅是材料种类。")
            else:
                safety = t(f"**For crew safety:** CO readings differ {ratio:.0f}× between materials, so material choice "
                           "matters for toxic-gas risk.",
                           f"**对乘员安全的意义：** 不同材料的 CO 读数相差 {ratio:.0f} 倍，因此材料选择对有毒气体风险很重要。")
        over = [r for r in rows if r["co_over_range_readings"]]
        if over:
            n_over = sum(r["co_over_range_readings"] for r in over)
            points.append(t(f"**{n_over} CO readings went above the sensor's range** ("
                            + ", ".join(f"{name(r)}: {r['co_over_range_readings']}" for r in over)
                            + "). They are left out of the averages, so those means are lower than the truth.",
                            f"**有 {n_over} 个 CO 读数超出传感器量程**（"
                            + "，".join(f"{name(r)}：{r['co_over_range_readings']}" for r in over)
                            + "）。它们未计入平均值，因此这些平均值偏低。"))
        if few:
            points.append(t(f"**Too few tests to rank reliably** (fewer than {MIN_TESTS_TO_RANK}): "
                            + ", ".join(f"{name(r)} ({r['empirical_test_count']})" for r in few) + ".",
                            f"**试验次数太少，无法可靠排名**（少于 {MIN_TESTS_TO_RANK} 次）："
                            + "，".join(f"{name(r)}（{r['empirical_test_count']} 次）" for r in few) + "。"))
        caveat = t("Final CO depends on sample size, airflow and burn time, so this compares the tests as they were run.",
                   "最终 CO 受样品尺寸、气流与燃烧时长影响，因此这里比较的是试验的实际情况。")
        return {"points": points, "safety": safety, "caveat": caveat}

    # FLEX droplets
    with_k = [r for r in rows if r["mean_burning_rate_constant_mm2_s"] is not None]
    if len(with_k) >= 2:
        fast, slow = with_k[0], with_k[-1]
        k1, k2 = fast["mean_burning_rate_constant_mm2_s"], slow["mean_burning_rate_constant_mm2_s"]
        points.append(t(f"**Fastest burning:** {fast['fuel_name']} droplets, K = **{k1:.3f} mm²/s**, about "
                        f"**{k1 / k2 - 1:.0%} faster** than {slow['fuel_name']} (K = {k2:.3f} mm²/s).",
                        f"**燃烧最快：** {fast['fuel_name']} 液滴，K = **{k1:.3f} mm²/s**，比 {slow['fuel_name']}"
                        f"（K = {k2:.3f} mm²/s）快约 **{k1 / k2 - 1:.0%}**。"))
    shares = [r for r in rows if r["extinction_share_of_non_disrupted"] is not None]
    if shares:
        def share(r: dict) -> str:
            c = r["outcome_counts"]
            return (f"{r['fuel_name']} **{r['extinction_share_of_non_disrupted']:.0%}** "
                    f"({c['Extinction']}/{c['Extinction'] + c['Completion']})")
        disrupted = ", ".join(f"{r['fuel_name']} {r['outcome_counts']['Disruption']}" for r in rows)
        points.append(t("**Flame went out before the droplet was used up:** " + " · ".join(share(r) for r in shares)
                        + f". Disrupted tests are left out ({disrupted}).",
                        "**液滴耗尽前火焰熄灭的比例：** " + " · ".join(share(r) for r in shares)
                        + f"。液滴破碎（中断）的试验不计入（{disrupted}）。"))
    sizes = [r for r in rows if r["mean_extinction_diameter_mm"] is not None]
    if sizes:
        points.append(t("**Droplet size when the flame went out** (mean): "
                        + " · ".join(f"{r['fuel_name']} {r['mean_extinction_diameter_mm']:.2f} mm "
                                     f"(start {r['mean_initial_diameter_mm']:.2f} mm)" for r in sizes) + ".",
                        "**火焰熄灭时的液滴直径**（平均）："
                        + " · ".join(f"{r['fuel_name']} {r['mean_extinction_diameter_mm']:.2f} mm"
                                     f"（初始 {r['mean_initial_diameter_mm']:.2f} mm）" for r in sizes) + "。"))
    ext = sum(r["outcome_counts"]["Extinction"] for r in rows)
    comp = sum(r["outcome_counts"]["Completion"] for r in rows)
    safety = t(f"**For crew safety:** most of these microgravity flames went out on their own ({ext} of {ext + comp}), "
               f"but **{comp} burned until the fuel was gone**, so self-extinction cannot be relied on.",
               f"**对乘员安全的意义：** 这些微重力火焰大多会自行熄灭（{ext + comp} 次中 {ext} 次），"
               f"但仍有 **{comp} 次一直燃烧到燃料耗尽**，因此不能依赖火焰自熄。") if ext + comp else ""
    caveat = t("These averages mix all FLEX atmospheres (different pressures and O₂, CO₂ and helium levels); the FLEX "
               "models in the Scientific Audit tab take those conditions into account.",
               "这些平均值混合了 FLEX 的所有气氛条件（不同的压力以及 O₂、CO₂、氦含量）；科学审计页中的 FLEX 模型考虑了这些条件。")
    return {"points": points, "safety": safety, "caveat": caveat}
