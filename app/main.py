from pydantic import BaseModel, Field

class ParametricSweepRequest(BaseModel):
    sweep_param: str = Field(default="flow_speed", description="扫描自变量: flow_speed, gravity, oxygen")
    range_min: float = Field(default=0.0, ge=0.0)
    range_max: float = Field(default=0.3, le=50.0)
    steps: int = Field(default=20, ge=5, le=100)
    fixed_gravity_m_s2: float = Field(default=0.0, ge=0.0, le=20.0)
    fixed_flow_speed_m_s: float = Field(default=0.05, ge=0.0, le=2.0)
    fixed_oxygen_percentage: float = Field(default=21.0, ge=12.0, le=50.0)

@app.get("/api/v1/research/empirical-ranking", tags=["science platform"])
def research_empirical_ranking(track_key: str = Query(default="solid_fuel")) -> dict[str, Any]:
    """100% 真实数据驱动：从 NASA 实验表中聚合统计材料火灾危险度与产物排名"""
    from app.services.research_engine import get_empirical_material_ranking
    return {"track_key": track_key, "ranking": get_empirical_material_ranking(track_key)}

@app.post("/api/v1/research/parametric-sweep", tags=["science platform"])
def research_parametric_sweep(req: ParametricSweepRequest) -> dict[str, Any]:
    """STEM 物理参数扫描：动态解算物理量曲线并输出机理解释叙述"""
    from app.services.research_engine import run_parametric_sensitivity_sweep
    return run_parametric_sensitivity_sweep(
        sweep_param=req.sweep_param,
        range_min=req.range_min,
        range_max=req.range_max,
        steps=req.steps,
        fixed_gravity_m_s2=req.fixed_gravity_m_s2,
        fixed_flow_speed_m_s=req.fixed_flow_speed_m_s,
        fixed_o2_pct=req.fixed_oxygen_percentage
    )