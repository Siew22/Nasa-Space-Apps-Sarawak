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

    for val in sweep_values:
        # 确定当前点参数
        g = val if sweep_param == "gravity" else fixed_gravity_m_s2
        u = val if sweep_param == "flow_speed" else fixed_flow_speed_m_s
        o2 = val if sweep_param == "oxygen" else fixed_o2_pct
        p = fixed_pressure_kpa

        u_safe = max(0.001, u)
        delta_t = 1200.0
        beta = 1.0 / 300.0

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

    # 生成面向学生的物理因果机制叙述 (Scientific Narration for STEM Students)
    narration = []
    if en:
        if sweep_param == "flow_speed":
            narration.append(
                "[Ventilation is a double-edged sword in space]: At microgravity (g=0) there is no buoyant "
                "convection to lift hot gas. When ventilation speed is near 0, the oxygen consumed by the flame "
                "cannot be replenished, and CO/CO2 accumulates around the fuel, so the flame suffocates and "
                "self-extinguishes (oxygen starvation). As ventilation increases, forced convection brings in "
                "fresh oxygen and flame spread accelerates markedly."
            )
        elif sweep_param == "gravity":
            narration.append(
                "[How gravity shapes the flame (Richardson number)]: At microgravity (g=0, log(Ri) < -1) the flow "
                "is controlled entirely by forced ventilation and the flame is spherical or blunt. As gravity rises "
                "to Earth normal (g=9.81 m/s2, log(Ri) > 1), buoyancy takes over: density differences accelerate "
                "hot gas upward and form the tall, tapered plume we know on the ground."
            )
        elif sweep_param == "oxygen":
            narration.append(
                "[Safety limits of oxygen-enriched cabins]: Crewed spacecraft and pre-EVA procedures often use "
                "reduced-pressure, oxygen-enriched atmospheres (e.g. 30-34% O2). Raising oxygen concentration "
                "widens the flammability limit exponentially, so a weak airflow that would barely sustain a flame "
                "in normal air can drive vigorous flame spread in an enriched atmosphere."
            )
    elif sweep_param == "flow_speed":
        narration.append(
            "【风速对太空燃烧的双刃剑效应】：在微重力（g=0）下，不存在热烟气上升的浮力对流。"
            "当通风风速接近 0 时，燃烧消耗掉的氧气无法及时通过气流补充，而产生的大量 CO/CO₂ 滞留在火源周围，"
            "火焰会被自身废气‘窒息’而自熄（Oxygen Starvation）。随着通风风速增加，强迫对流带入新鲜氧气，蔓延速率显著加快。"
        )
    elif sweep_param == "gravity":
        narration.append(
            "【重力对火焰形态的支配规律（理查森数演化）】：在微重力下（g=0, log(Ri) < -1），流动完全由强迫通风控制，火焰呈球形或钝头；"
            "当重力逐渐增加至地球常重力（g=9.81 m/s², log(Ri) > 1）时，浮力效应急剧占据主导地位，"
            "热废气因密度差被强烈抽吸向上加速，形成我们地面上熟悉的锥形狭长羽流。"
        )
    elif sweep_param == "oxygen":
        narration.append(
            "【探索舱富氧气氛的安全临界】：载人航天登月舱或出舱活动前常使用低压富氧环境（如 30%~34% O₂）。"
            "计算表明，提高氧浓度会按指数拓宽可燃极限，原本在常态空气下微弱的通风气流，在富氧下即可引起剧烈的火焰蔓延。"
        )

    return {
        "sweep_parameter": sweep_param,
        "curve_points": curve_points,
        "scientific_narration": narration
    }
