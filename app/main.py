from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.settings import allowed_origins

app = FastAPI(title="Flame in Freefall API")

# Allow browser frontends (Streamlit or a separate SPA) to call this API from
# their deployed origins. Origins come from the FRONTEND_ORIGINS env var;
# see app/settings.py for the localhost defaults.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

@app.get("/api/v1/research/provenance", tags=["science platform"])
def research_provenance() -> dict[str, Any]:
    """Status of every dataset as computed by src/provenance_audit.py (never assumed)."""
    from app.services.provenance import STATUS_VOCABULARY, dataset_status, load_report

    report = load_report()
    keys = sorted(report["datasets"]) if report else ["BASS_II", "FLEX", "MACFP_PMMA_CONE"]
    return {"status_vocabulary": STATUS_VOCABULARY,
            "report_available": report is not None,
            "datasets": [dataset_status(k, report) for k in keys]}


@app.get("/api/v1/research/empirical-ranking", tags=["science platform"])
def research_empirical_ranking(track_key: str = Query(default="solid_fuel")) -> dict[str, Any]:
    """Per-material aggregates computed from the loaded BASS-II / FLEX-format table.

    `data_status` says what that table really is; in this build it is synthetic demo data,
    so the ranking demonstrates the computation and is not a scientific result.
    """
    from app.services.provenance import dataset_status
    from app.services.research_engine import get_empirical_material_ranking

    status = dataset_status("BASS_II" if track_key == "solid_fuel" else "FLEX")
    return {"track_key": track_key, "data_status": status["status"], "result_type": status["status"],
            "warning": status["warning"], "ranking": get_empirical_material_ranking(track_key)}


@app.post("/api/v1/research/parametric-sweep", tags=["science platform"])
def research_parametric_sweep(req: ParametricSweepRequest) -> dict[str, Any]:
    """Educational parameter sweep from first-principles scaling formulas (a simulation, not data)."""
    from app.services.research_engine import run_parametric_sensitivity_sweep
    result = run_parametric_sensitivity_sweep(
        sweep_param=req.sweep_param,
        range_min=req.range_min,
        range_max=req.range_max,
        steps=req.steps,
        fixed_gravity_m_s2=req.fixed_gravity_m_s2,
        fixed_flow_speed_m_s=req.fixed_flow_speed_m_s,
        fixed_o2_pct=req.fixed_oxygen_percentage
    )
    return {"result_type": "SIMULATED_RESULT", **result}


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

    from app.services.provenance import dataset_status

    try:
        predictions = predict_fire_rows([row.model_dump() for row in req.rows])
        model_info = get_model_info()
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    training = dataset_status("MACFP_PMMA_CONE")
    return {
        "result_type": "MODEL_PREDICTION",
        "model_id": model_info["model_id"],
        "source": model_info["source"],
        "scope": model_info["scope"],
        "training_data_status": training["status"],
        "training_data_warning": training["warning"],
        "validation": model_info["validation"],
        "validation_note": ("'validation' is the single 5-fold run stored with the model. The repeated, "
                            "leakage-audited evaluation is in reports/cv_summary.json (task macfp_pmma_peak_hrr)."),
        "hazard_definition": "Project-defined 350 kW/m^2 threshold applied to predicted peak HRR; not an independent classifier or fire-safety determination.",
        "predictions": predictions,
    }
