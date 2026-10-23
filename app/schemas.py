from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.settings import read_json_config

_MODEL_CONFIG = read_json_config()
_DEFAULTS = _MODEL_CONFIG["constants"]
_RANGES = _MODEL_CONFIG["parameter_ranges"]


def _minimum(name: str) -> float:
    return float(_RANGES[name]["minimum"])


def _exclusive_minimum(name: str) -> float:
    return float(_RANGES[name]["exclusive_minimum"])


def _maximum(name: str) -> float:
    return float(_RANGES[name]["maximum"])


class SimulationRequest(BaseModel):
    """Inputs for an explicitly educational, reduced-order 2D model.

    `steps` and `fps` are alternative ways to set temporal resolution. If fps is
    supplied, steps is derived from duration * fps. All numerical bounds come
    from config/simulation_model.json where applicable.
    """

    model_config = ConfigDict(extra="forbid")

    effective_gravity_m_s2: float = Field(
        default_factory=lambda: float(_DEFAULTS["standard_earth_gravity_m_s2"]),
        ge=_minimum("effective_gravity_m_s2"), le=_maximum("effective_gravity_m_s2"),
        description="Effective gravitational acceleration; 0 is an idealized teaching boundary, not a complete ISS acceleration history.",
    )
    ambient_temperature_k: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_ambient_temperature_k"]),
        ge=_minimum("ambient_temperature_k"), le=_maximum("ambient_temperature_k"),
    )
    hot_gas_temperature_k: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_hot_gas_temperature_k"]),
        ge=_minimum("hot_gas_temperature_k"), le=_maximum("hot_gas_temperature_k"),
    )
    pressure_pa: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_pressure_pa"]),
        ge=_minimum("pressure_pa"), le=_maximum("pressure_pa"),
    )
    parcel_volume_m3: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_parcel_volume_m3"]),
        gt=_exclusive_minimum("parcel_volume_m3"), le=_maximum("parcel_volume_m3"),
    )
    length_scale_m: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_length_scale_m"]),
        gt=_exclusive_minimum("length_scale_m"), le=_maximum("length_scale_m"),
    )
    forced_flow_speed_m_s: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_forced_flow_speed_m_s"]),
        ge=_minimum("forced_flow_speed_m_s"), le=_maximum("forced_flow_speed_m_s"),
        description="Imposed horizontal flow speed used in this reduced-order illustration.",
    )
    diffusion_coefficient_m2_s: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_diffusion_coefficient_m2_s"]),
        gt=_exclusive_minimum("diffusion_coefficient_m2_s"), le=_maximum("diffusion_coefficient_m2_s"),
    )
    duration_s: float = Field(
        default_factory=lambda: float(_DEFAULTS["default_duration_s"]),
        gt=_exclusive_minimum("duration_s"), le=_maximum("duration_s"),
    )
    steps: int | None = Field(default=None, ge=int(_RANGES["steps"]["minimum"]), le=int(_RANGES["steps"]["maximum"]))
    fps: int | None = Field(default=None, ge=int(_RANGES["fps"]["minimum"]), le=int(_RANGES["fps"]["maximum"]))
    grid_resolution: int = Field(
        default_factory=lambda: int(_DEFAULTS["default_grid_resolution"]),
        ge=int(_RANGES["grid_resolution"]["minimum"]),
        le=int(_RANGES["grid_resolution"]["maximum"]),
    )

    @model_validator(mode="after")
    def validate_physics_and_resource_budget(self) -> "SimulationRequest":
        if self.hot_gas_temperature_k <= self.ambient_temperature_k:
            raise ValueError("hot_gas_temperature_k must be greater than ambient_temperature_k for this buoyancy example")

        if self.steps is not None and self.fps is not None:
            raise ValueError("Provide either 'steps' or 'fps', not both; when fps is supplied, steps is derived from duration_s * fps.")

        if self.fps is not None:
            effective_steps = int(round(self.duration_s * self.fps))
        elif self.steps is not None:
            effective_steps = self.steps
        else:
            effective_steps = int(_DEFAULTS["default_steps"])
        if not int(_RANGES["steps"]["minimum"]) <= effective_steps <= int(_RANGES["steps"]["maximum"]):
            raise ValueError(
                f"Effective step count {effective_steps} is outside the configured range "
                f"[{_RANGES['steps']['minimum']}, {_RANGES['steps']['maximum']}]. Reduce duration_s/fps or set steps directly."
            )
        self.steps = effective_steps

        cfg_budget = int(_DEFAULTS["max_grid_cell_frames"])
        cell_frames = (self.steps + 1) * self.grid_resolution * self.grid_resolution
        if cell_frames > cfg_budget:
            raise ValueError(
                f"Requested grid/time resolution needs {cell_frames} grid-cell frames, exceeding the configured budget of {cfg_budget}. "
                "Reduce steps/fps, grid_resolution, or duration_s."
            )
        return self


class SimulationPoint(BaseModel):
    time_s: float
    diffusion_length_scale_m: float
    buoyancy_force_n: float
    hot_gas_weight_n: float
    net_buoyancy_force_n: float
    reduced_gravity_m_s2: float
    buoyancy_velocity_scale_m_s: float
    forced_flow_speed_m_s: float


class SimulationGrid2D(BaseModel):
    x_coords_m: list[float] = Field(..., description="X coordinates shared by each row of the grid, in metres.")
    y_coords_m: list[float] = Field(..., description="Y coordinates shared by each column of the grid, in metres.")
    temperature_grid_k: list[list[float]] = Field(..., description="Schematic temperature scalar field [Ny, Nx], in kelvin.")
    velocity_u_grid_m_s: list[list[float]] = Field(..., description="Illustrative horizontal velocity component [Ny, Nx], in m/s.")
    velocity_v_grid_m_s: list[list[float]] = Field(..., description="Illustrative vertical velocity component [Ny, Nx], in m/s.")


class SimulationFrame(BaseModel):
    time_s: float
    step_index: int
    plume_centroid_m: list[float] = Field(..., description="Illustrative parcel-centroid coordinate [x, y] in metres.")
    diffusion_radius_m: float = Field(..., description="Characteristic width of the schematic Gaussian field; not a measured flame radius.")
    net_buoyancy_force_n: float
    spatial_grid: SimulationGrid2D


class SimulationResponse(BaseModel):
    model_id: str
    model_version: str
    model_type: str
    scientific_status: str
    inputs: dict[str, float | int]
    derived_values: dict[str, float | None]
    time_series: list[SimulationPoint]
    frames: list[SimulationFrame] = Field(..., description="Generated educational time frames for 2D rendering; not empirical NASA images.")
    grid_metadata: dict[str, Any]
    equations: list[dict[str, str]]
    assumptions: list[str]
    limitations: list[str]
    references: list[str]
