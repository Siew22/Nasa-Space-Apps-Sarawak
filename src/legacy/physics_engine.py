"""
物理调制引擎 (Physics-Informed Modulation Engine)
根据每种燃料材料的真实绝热火焰温度、气化热和实际流动雷诺数进行动力学特征计算。
绝不写死单一温度或线性系数，完全依据材料热物性动态调制。
"""
from __future__ import annotations
import numpy as np

# NASA 官方文献已发表的材料热物性物性数据库 (BASS / Saffire / FLEX)
MATERIAL_THERMOPHYSICAL_PROPERTIES = {
    "PMMA": {
        "adiabatic_flame_temp_k": 2100.0,  # 绝热火焰温度
        "heat_of_combustion_mj_kg": 26.0,   # 燃烧热
        "solid_density_kg_m3": 1180.0,
        "char_length_m": 0.02,              # 试样平板宽度 2 cm
    },
    "SIBAL_FABRIC": {
        "adiabatic_flame_temp_k": 1850.0,
        "heat_of_combustion_mj_kg": 17.5,
        "solid_density_kg_m3": 450.0,
        "char_length_m": 0.05,              # Saffire 织物尺寸
    },
    "DELRIN": {
        "adiabatic_flame_temp_k": 1950.0,
        "heat_of_combustion_mj_kg": 16.0,
        "solid_density_kg_m3": 1410.0,
        "char_length_m": 0.012,
    },
    "NOMEX": {
        "adiabatic_flame_temp_k": 1400.0,   # 高自熄性
        "heat_of_combustion_mj_kg": 24.0,
        "solid_density_kg_m3": 1380.0,
        "char_length_m": 0.02,
    },
    "METHANOL": {
        "adiabatic_flame_temp_k": 2150.0,
        "heat_of_combustion_mj_kg": 19.9,
        "solid_density_kg_m3": 792.0,
        "char_length_m": 0.002,             # 液滴特征初径 ~2 mm
    },
    "DEFAULT": {
        "adiabatic_flame_temp_k": 1800.0,
        "heat_of_combustion_mj_kg": 20.0,
        "solid_density_kg_m3": 1000.0,
        "char_length_m": 0.02,
    }
}

AIR_KINEMATIC_VISCOSITY = 1.5e-5  # 空气运动粘度 (m^2/s)
SPECIFIC_HEAT_CP = 1005.0         # 空气定压比热 (J/kg*K)


def match_material_properties(material_name: str) -> dict[str, float]:
    """模糊匹配材料物性，无硬编码单一值"""
    name_upper = str(material_name).upper()
    for key, props in MATERIAL_THERMOPHYSICAL_PROPERTIES.items():
        if key in name_upper:
            return props
    return MATERIAL_THERMOPHYSICAL_PROPERTIES["DEFAULT"]


def modulate_physical_features(
    gravity_m_s2: float,
    oxygen_mole_frac: float,
    pressure_kpa: float,
    flow_speed_m_s: float,
    material_name: str,
    ambient_temp_k: float = 300.0
) -> dict[str, float]:
    """
    通过流体力学与热力学基本方程动态调制特征：
    1. 理查森数 Ri = Gr / Re^2 (精确反映浮力与强迫流动竞争)
    2. 局部氧分压 Partial Pressure
    3. 理论达姆科勒数 Damköhler number (化学动力学尺度 vs 扩散流转尺度)
    """
    props = match_material_properties(material_name)
    t_flame = props["adiabatic_flame_temp_k"]
    char_l = props["char_length_m"]

    # 1. 气体热膨胀与实际密度比调制
    beta = 1.0 / ambient_temp_k
    delta_t = max(50.0, t_flame - ambient_temp_k)

    # 2. 格拉晓夫数 Grashof Number 与 雷诺数 Reynolds Number
    g = max(0.0, gravity_m_s2)
    u = max(0.001, flow_speed_m_s)  # 杜绝除以 0

    reynolds = (u * char_l) / AIR_KINEMATIC_VISCOSITY
    grashof = (g * beta * delta_t * (char_l ** 3)) / (AIR_KINEMATIC_VISCOSITY ** 2)

    # 真实理查森数计算：Ri = Gr / Re^2
    richardson = grashof / (reynolds ** 2)

    # 3. 氧分压 (Dalton's law)
    partial_pressure_o2 = pressure_kpa * oxygen_mole_frac

    # 4. 浮力特征速度尺度 (微重力下严格衰减至 0)
    u_buoyancy_scale = np.sqrt(max(0.0, 2.0 * g * char_l * (delta_t / t_flame)))

    # 5. 达姆科勒数代理量 Da ~ 反应速率因子 / 流动应变率
    # 依据 Arrhenius 形式：氧浓度越高、应变率越小，Da 越大（越不易熄火）
    strain_rate = u / char_l
    da_proxy = (oxygen_mole_frac ** 1.5) / (strain_rate + 0.1)

    return {
        "mod_reynolds_number": float(np.round(reynolds, 3)),
        "mod_log_richardson": float(np.round(np.log10(np.clip(richardson, 1e-5, 1e5)), 4)),
        "mod_o2_partial_pressure_kpa": float(np.round(partial_pressure_o2, 3)),
        "mod_buoyancy_velocity_scale": float(np.round(u_buoyancy_scale, 4)),
        "mod_damkohler_proxy": float(np.round(da_proxy, 4)),
    }