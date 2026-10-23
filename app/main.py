from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = FastAPI(title="Flame in Freefall API")


class FirePredictionRow(BaseModel):
    """Numbers measured after a burn, in the units named by each field."""

    model_config = ConfigDict(extra="forbid")

    radiant_heat_flux_kw_m2: float = Field(ge=0, allow_inf_nan=False)
    real_burn_duration_s: float = Field(gt=0, allow_inf_nan=False)
    real_mean_hrr_kw_m2: float = Field(gt=0, allow_inf_nan=False)

    @field_validator(
        "radiant_heat_flux_kw_m2", "real_burn_duration_s", "real_mean_hrr_kw_m2", mode="before"
    )
    @classmethod
    def require_json_number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a JSON number")
        return value


class FirePredictionBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[FirePredictionRow] = Field(min_length=1, max_length=1000)


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


@app.post("/api/v1/research/predict-fire", tags=["science platform"])
def research_predict_fire(req: FirePredictionBatchRequest) -> dict[str, Any]:
    """Predict peak heat release from post-test burn summaries in row order.

    The high-hazard flag applies the project's threshold to the predicted peak.
    It is not a separately trained classifier or a fire-safety determination.
    """
    from app.services.model_inference import (
        ModelUnavailableError,
        get_model_info,
        predict_fire_rows,
    )

    try:
        predictions = predict_fire_rows([row.model_dump() for row in req.rows])
        model_info = get_model_info()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_id": model_info["model_id"],
        "source": model_info["source"],
        "scope": model_info["scope"],
        "validation": model_info["validation"],
        "hazard_definition": "Project-defined 350 kW/m^2 threshold applied to predicted peak HRR; not an independent classifier or fire-safety determination.",
        "predictions": predictions,
    }
