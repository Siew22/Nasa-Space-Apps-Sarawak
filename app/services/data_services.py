from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# 严格定义的科学分轨物理字段字典 (规范物理含义与标准单位，防止混淆)
TRACK_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "solid_fuel": {
        "track_name": "固体表面燃烧与微重力灭火 (Solid Fuel Combustion & Suppression)",
        "physical_phenomenon": "固体聚合物热解、顺流/逆流火焰蔓延、气流熄火边界",
        "primary_dataset": "BASS_II",
        "reference_doi": "10.60555/4qc4-de67",
        "fields": {
            "fuel_sample_material": {"label": "燃料材料类型", "unit": "text", "role": "material_spec"},
            "fan_display": {"label": "风扇指示读数 (关联风速)", "unit": "display_units", "role": "flow_condition"},
            "calibrated_initial_o2_pct_by_vol": {"label": "初始标定氧浓度", "unit": "% Vol", "role": "environmental_condition"},
            "calibrated_final_o2_pct_by_vol": {"label": "结束标定氧浓度", "unit": "% Vol", "role": "environmental_response"},
            "initial_co_ppm": {"label": "初始一氧化碳浓度", "unit": "ppm", "role": "gas_generation"},
            "final_co_ppm": {"label": "结束一氧化碳浓度", "unit": "ppm", "role": "gas_generation"},
            "total_frames_shot": {"label": "拍摄总帧数", "unit": "frames", "role": "measurement_record"}
        }
    },
    "liquid_droplet": {
        "track_name": "空间悬浮液滴燃烧与冷焰熄灭 (Liquid Droplet Combustion & Cold Flame)",
        "physical_phenomenon": "球对称扩散蒸发燃烧、d^2定律衰减、辐射熄灭直径",
        "primary_dataset": "FLEX",
        "reference_doi": "10.60555/mbq8-0451",
        "fields": {
            "fuel": {"label": "液体燃料品类", "unit": "chemical_formula", "role": "material_spec"},
            "ambient_pressure_mmhg": {"label": "环境绝对压力", "unit": "mmHg", "role": "environmental_condition"},
            "o_initial_ambient_composition_mole_fraction": {"label": "初始氧气摩尔分数", "unit": "mole_fraction", "role": "environmental_condition"},
            "droplet_initial_diameter_mm": {"label": "液滴初始几何直径 (d0)", "unit": "mm", "role": "initial_geometry"},
            "visible_flame_extinction_diameter_mm": {"label": "可见火焰熄灭直径 (d_ext)", "unit": "mm", "role": "extinction_measurement"},
            "burning_rate_mm": {"label": "液滴有效线消耗速度", "unit": "mm/s", "role": "burning_kinetics"},
            "burn_time_s": {"label": "持续燃烧时间", "unit": "s", "role": "combustion_duration"},
            "test_end": {"label": "实验终态判定 (Extinction / Disruption)", "unit": "categorical", "role": "outcome_state"}
        }
    }
}


def list_tracks() -> dict[str, Any]:
    """返回所有支持的物理分轨目录"""
    return {
        "available_tracks": list(TRACK_FIELD_SPECS.keys()),
        "tracks": TRACK_FIELD_SPECS
    }


def list_fields(dataset_key: str | None = None, track_key: str | None = None) -> list[dict[str, Any]]:
    """支持按 track_key 或 dataset_key 过滤字段元数据"""
    fld_path = PROCESSED_DIR / "field_catalog.csv"
    if not fld_path.exists():
        return []
    df = pd.read_csv(fld_path)
    
    # 若指定物理轨道，则对齐其专属数据集
    if track_key == "solid_fuel":
        dataset_key = "BASS_II"
    elif track_key == "liquid_droplet":
        dataset_key = "FLEX"

    if dataset_key:
        df = df[df["dataset_key"].astype(str) == dataset_key]
        
    return df.fillna("").to_dict(orient="records")


def list_records(
    dataset_key: str | None = None,
    track_key: str | None = None,
    search: str | None = None,
    offset: int = 0,
    limit: int = 100
) -> tuple[int, list[dict[str, Any]]]:
    rec_path = PROCESSED_DIR / "dashboard_records.csv"
    if not rec_path.exists():
        return 0, []
    df = pd.read_csv(rec_path, low_memory=False)

    # 物理分轨逻辑过滤
    if track_key == "solid_fuel":
        df = df[df["dataset_key"].astype(str) == "BASS_II"]
    elif track_key == "liquid_droplet":
        df = df[df["dataset_key"].astype(str) == "FLEX"]

    if dataset_key:
        df = df[df["dataset_key"].astype(str) == dataset_key]

    if search:
        s = search.lower()
        mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(s, regex=False, na=False)).any(axis=1)
        df = df[mask]

    total = len(df)
    subset = df.iloc[offset: offset + limit].fillna("")
    return total, subset.to_dict(orient="records")


def get_record_by_id(record_id: str) -> dict[str, Any] | None:
    rec_path = PROCESSED_DIR / "dashboard_records.csv"
    if not rec_path.exists():
        return None
    df = pd.read_csv(rec_path, low_memory=False)
    res = df[df["record_id"] == record_id]
    return res.iloc[0].fillna("").to_dict() if not res.empty else None


def get_dataset_catalog() -> list[dict[str, Any]]:
    cat_path = PROCESSED_DIR / "dataset_catalog.csv"
    if not cat_path.exists():
        return []
    return pd.read_csv(cat_path).fillna("").to_dict(orient="records")


def dataset_summary(dataset_key: str | None = None) -> dict[str, Any]:
    sum_path = PROCESSED_DIR / "dataset_summary.csv"
    if not sum_path.exists():
        return {"total_datasets": 0, "total_records": 0}
    df = pd.read_csv(sum_path)
    if dataset_key:
        df = df[df["dataset_key"].astype(str) == dataset_key]
    return {"total_records": int(df["record_count"].sum()) if not df.empty else 0}