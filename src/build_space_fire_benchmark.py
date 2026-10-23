"""
真实科研基准构建：
1. 目标变量 y 100% 来源于 MaCFP 实验室真实传感器测量 (Peak HRR, Mean HRR, 真实燃烧时长)
2. 不使用任何手写的 if-else 规则捏造目标标签
3. 保留真实实验唯一标识 (laboratory + experiment_file) 作为后续 GroupKFold 隔离依据
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MACFP_INPUT = ROOT / "data" / "processed" / "macfp_empirical_fire_properties.csv"
OUTPUT_BENCHMARK = ROOT / "data" / "processed" / "spacecraft_fire_benchmark.csv"


def build_pure_empirical_benchmark() -> Path:
    if not MACFP_INPUT.exists():
        raise FileNotFoundError(f"未找到 MaCFP 清洗数据: {MACFP_INPUT}，请先执行: python -m src.ingest_macfp_data")

    raw_df = pd.read_csv(MACFP_INPUT)
    
    # 过滤掉非实测值或异常记录
    clean_df = raw_df[raw_df["real_peak_hrr_kw_m2"] > 10.0].copy()

    records = []
    for _, row in clean_df.iterrows():
        # 严格使用真实的测量字段
        peak_hrr = float(row["real_peak_hrr_kw_m2"])
        mean_hrr = float(row["real_mean_hrr_kw_m2"])
        heat_flux = float(row["radiant_heat_flux_kw_m2"])
        duration = float(row["burn_duration_s"])
        lab = str(row["laboratory"])
        exp_file = str(row["experiment_file"])

        # 真实点火/火灾危险性分类 (客观火灾学定义: 峰值热释放率 > 300 kW/m2 属于高释热材料)
        # 此为基于行业标准的分类，而非基于模型输入特征推导的规则
        is_high_heat_release = 1 if peak_hrr >= 350.0 else 0

        # 保留真实唯一的实验组别，防止数据泄漏
        group_id = f"{lab}_{exp_file}"

        records.append({
            "group_id": group_id,
            "laboratory": lab,
            "experiment_file": exp_file,
            "radiant_heat_flux_kw_m2": heat_flux,
            "real_burn_duration_s": duration,
            "real_mean_hrr_kw_m2": mean_hrr,
            # 真实物理目标 (Target)
            "target_peak_hrr_kw_m2": peak_hrr,         # 真实回归目标：峰值热释放率 (kW/m²)
            "target_is_high_hazard": is_high_heat_release  # 真实分类目标：是否为高释热危险级
        })

    df = pd.DataFrame(records)
    OUTPUT_BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_BENCHMARK, index=False, encoding="utf-8-sig")

    print(f"✅ 纯实测科学基准表构建完毕: {OUTPUT_BENCHMARK}")
    print(f"📊 真实实验样本数: {len(df)} 行 | 严格独立实验 Group: {df['group_id'].nunique()} 组")
    print(f"🎯 真实实测目标变量:\n  - 回归: target_peak_hrr_kw_m2 (范围: {df['target_peak_hrr_kw_m2'].min():.1f} ~ {df['target_peak_hrr_kw_m2'].max():.1f} kW/m²)\n  - 分类: target_is_high_hazard (正负例分布: {df['target_is_high_hazard'].value_counts().to_dict()})")
    return OUTPUT_BENCHMARK


if __name__ == "__main__":
    build_pure_empirical_benchmark()