"""
Phase 3 物理引擎边界自洽性测试 (Physics Sanity & Boundary Consistency Tests):
1. 微重力极限检验: g=0 时垂直浮力加速度、上升位移与垂直速度恒等于 0
2. 重力尺度单调性: 浮力特征速度尺度严格随 sqrt(g) 单调递增
3. 羽流偏转物理规律: 地球表面浮力强烈使羽流偏竖直，月球低重力使羽流顺风倒伏
4. 时间与时空连续性: t=0 时质心与扩散半径连续平滑，无突变跳跃
5. 负浮力物理保护: 严禁 hot_gas_temperature_k <= ambient_temperature_k
"""
import math
import pytest
from app.schemas import SimulationRequest
from app.services.simulation_service import run_simulation


def test_microgravity_limit_zero_buoyancy_motion():
    """验证 g=0 微重力极限: 质心在垂直方向完全静止，浮力为零"""
    req = SimulationRequest(
        effective_gravity_m_s2=0.0,
        forced_flow_speed_m_s=0.05,
        duration_s=2.0,
        steps=20,
        grid_resolution=15
    )
    res = run_simulation(req)

    # 1. 守恒量断言
    assert res.derived_values["reduced_gravity_m_s2"] == 0.0
    assert res.derived_values["net_buoyancy_force_n"] == 0.0
    assert res.derived_values["buoyancy_velocity_scale_m_s"] == 0.0

    # 2. 帧序列空间轨迹断言
    for frame in res.frames:
        x_c, y_c = frame.plume_centroid_m
        # 垂直位置必须恒为 0 (完全不向上飘)
        assert math.isclose(y_c, 0.0, abs_tol=1e-5), f"微重力下垂直位移异常: {y_c}"
        # 垂直浮力速度场必须全为 0
        v_grid = frame.spatial_grid.velocity_v_grid_m_s
        max_v = max(max(abs(val) for val in row) for row in v_grid)
        assert math.isclose(max_v, 0.0, abs_tol=1e-5), f"微重力下垂直速度异常: {max_v}"


def test_gravity_monotonic_buoyancy_scaling():
    """验证重力单调性: 特征速度与净浮力严格满足 u_b(Earth) > u_b(Mars) > u_b(Moon) > u_b(0)"""
    gravity_scenarios = [0.0, 1.62, 3.71, 9.80665]
    u_scales = []
    net_forces = []

    for g in gravity_scenarios:
        req = SimulationRequest(
            effective_gravity_m_s2=g,
            duration_s=1.0,
            steps=5,
            grid_resolution=10
        )
        res = run_simulation(req)
        u_scales.append(res.derived_values["buoyancy_velocity_scale_m_s"])
        net_forces.append(res.derived_values["net_buoyancy_force_n"])

    # 严格单调递增断言
    assert u_scales[0] == 0.0
    assert u_scales[0] < u_scales[1] < u_scales[2] < u_scales[3]
    assert net_forces[0] == 0.0
    assert net_forces[0] < net_forces[1] < net_forces[2] < net_forces[3]


def test_plume_deflection_angle_comparison():
    """验证重力羽流偏转角: 在相同的水平风速下，地球上的浮力上升角明显大于月球"""
    flow_speed = 0.05

    # 月球情景
    res_moon = run_simulation(SimulationRequest(
        effective_gravity_m_s2=1.62,
        forced_flow_speed_m_s=flow_speed,
        duration_s=2.0, steps=10, grid_resolution=10
    ))
    # 地球情景
    res_earth = run_simulation(SimulationRequest(
        effective_gravity_m_s2=9.80665,
        forced_flow_speed_m_s=flow_speed,
        duration_s=2.0, steps=10, grid_resolution=10
    ))

    # 取最后一帧的质心位置计算偏转角 theta = arctan(y_c / x_c)
    x_moon, y_moon = res_moon.frames[-1].plume_centroid_m
    x_earth, y_earth = res_earth.frames[-1].plume_centroid_m

    theta_moon = math.atan2(y_moon, x_moon)
    theta_earth = math.atan2(y_earth, x_earth)

    # 地球上的浮力更强，火焰更竖立 (偏转角更大)
    assert theta_earth > theta_moon, f"地球羽流偏转角 ({theta_earth:.3f}) 应当大于月球 ({theta_moon:.3f})"


def test_temporal_continuity_and_smooth_centroid():
    """验证时间演化连续性: t=0 时位移为 0，扩散半径连续扩展"""
    req = SimulationRequest(
        effective_gravity_m_s2=9.81,
        forced_flow_speed_m_s=0.1,
        duration_s=1.0,
        steps=10,
        grid_resolution=10
    )
    res = run_simulation(req)

    # t=0 时质心必须从原点出发
    assert res.frames[0].time_s == 0.0
    assert res.frames[0].plume_centroid_m == [0.0, 0.0]

    # 扩散半径必须随时间严格单调增大 (r ~ sqrt(2Dt))
    radii = [f.diffusion_radius_m for f in res.frames]
    for i in range(len(radii) - 1):
        assert radii[i] < radii[i + 1]


def test_unphysical_temperature_rejected():
    """验证物理安全防线: 热气体温度必须高于环境温度，严禁输入负浮力或逆温"""
    with pytest.raises(ValueError, match="hot_gas_temperature_k must be greater than ambient_temperature_k"):
        SimulationRequest(
            ambient_temperature_k=300.0,
            hot_gas_temperature_k=250.0  # 逆温输入，必须被 Pydantic 拒绝
        )