from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.schemas import (
    SimulationFrame,
    SimulationGrid2D,
    SimulationPoint,
    SimulationRequest,
    SimulationResponse,
)
from app.settings import read_json_config


def _round_grid(values: np.ndarray, decimals: int = 6) -> list[list[float]]:
    """Convert a NumPy field into JSON-friendly nested lists."""
    return np.round(values, decimals=decimals).astype(float).tolist()


def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Return reduced-order physical scales and schematic 2D fields.

    This is deliberately not described as a flame/combustion CFD solver. The
    Gaussian scalar field and damped parcel-centroid closure are a deterministic
    educational illustration, not experimental output or a calibrated predictor.
    """
    config = read_json_config()
    constants = config["constants"]
    gas_constant = float(constants["specific_gas_constant_dry_air_j_kg_k"])
    radius_fraction = float(constants["initial_diffusion_radius_fraction_of_length_scale"])
    padding_sigma = float(constants["grid_padding_sigma"])

    rho_ambient = request.pressure_pa / (gas_constant * request.ambient_temperature_k)
    rho_hot = request.pressure_pa / (gas_constant * request.hot_gas_temperature_k)
    gravity = request.effective_gravity_m_s2
    volume = request.parcel_volume_m3

    buoyancy_force = rho_ambient * volume * gravity
    hot_weight = rho_hot * volume * gravity
    net_buoyancy_force = buoyancy_force - hot_weight
    reduced_gravity = (
        gravity * (rho_ambient - rho_hot) / rho_ambient
        if rho_ambient > 0.0
        else 0.0
    )
    # Input validation assures hot gas is warmer than ambient, so this should be >= 0.
    reduced_gravity = max(0.0, reduced_gravity)
    length_scale = request.length_scale_m
    buoyancy_velocity_scale = math.sqrt(reduced_gravity * length_scale)
    buoyancy_time_scale = (
        math.sqrt(length_scale / reduced_gravity) if reduced_gravity > 0.0 else None
    )
    diffusion_time_scale = length_scale**2 / request.diffusion_coefficient_m2_s
    buoyancy_to_flow_ratio = (
        buoyancy_velocity_scale / request.forced_flow_speed_m_s
        if request.forced_flow_speed_m_s > 0.0
        else None
    )

    n_steps = int(request.steps or constants["default_steps"])
    times = np.linspace(0.0, request.duration_s, n_steps + 1, dtype=np.float64)
    diffusion_radii = np.sqrt(
        (length_scale * radius_fraction) ** 2
        + 2.0 * request.diffusion_coefficient_m2_s * times
    )

    # A bounded, characteristic parcel-motion closure rather than constant
    # acceleration for all time. It is illustrative, not calibrated plume physics.
    if reduced_gravity > 0.0 and buoyancy_time_scale is not None:
        tau = buoyancy_time_scale
        vertical_velocity = buoyancy_velocity_scale * (1.0 - np.exp(-times / tau))
        vertical_centroid = buoyancy_velocity_scale * (
            times - tau * (1.0 - np.exp(-times / tau))
        )
    else:
        vertical_velocity = np.zeros_like(times)
        vertical_centroid = np.zeros_like(times)

    horizontal_centroid = request.forced_flow_speed_m_s * times

    # Domain is computed from requested motion and diffusion parameters so the
    # moving illustrative field remains inside the grid rather than leaving a
    # fixed hard-coded viewport.
    padding_m = padding_sigma * float(np.max(diffusion_radii))
    x_min = min(0.0, float(np.min(horizontal_centroid))) - padding_m
    x_max = max(0.0, float(np.max(horizontal_centroid))) + padding_m
    y_min = min(0.0, float(np.min(vertical_centroid))) - padding_m
    y_max = max(0.0, float(np.max(vertical_centroid))) + padding_m
    if math.isclose(x_min, x_max):
        x_min, x_max = x_min - padding_m, x_max + padding_m
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - padding_m, y_max + padding_m

    x_coords = np.linspace(x_min, x_max, request.grid_resolution, dtype=np.float64)
    y_coords = np.linspace(y_min, y_max, request.grid_resolution, dtype=np.float64)
    xx, yy = np.meshgrid(x_coords, y_coords)

    delta_temperature = request.hot_gas_temperature_k - request.ambient_temperature_k
    frames: list[SimulationFrame] = []
    time_series: list[SimulationPoint] = []
    for index, t_value in enumerate(times):
        t = float(t_value)
        x_c = float(horizontal_centroid[index])
        y_c = float(vertical_centroid[index])
        r_diff = float(diffusion_radii[index])
        distance_squared = (xx - x_c) ** 2 + (yy - y_c) ** 2
        gaussian = np.exp(-distance_squared / (2.0 * r_diff**2))

        # These fields are schematic renderable data; they are not an actual
        # numerical solution of the energy or momentum equations.
        temperature_field = request.ambient_temperature_k + delta_temperature * gaussian
        horizontal_velocity_field = np.full_like(xx, request.forced_flow_speed_m_s)
        vertical_velocity_field = float(vertical_velocity[index]) * gaussian

        grid = SimulationGrid2D(
            x_coords_m=np.round(x_coords, 6).astype(float).tolist(),
            y_coords_m=np.round(y_coords, 6).astype(float).tolist(),
            temperature_grid_k=_round_grid(temperature_field, decimals=3),
            velocity_u_grid_m_s=_round_grid(horizontal_velocity_field, decimals=6),
            velocity_v_grid_m_s=_round_grid(vertical_velocity_field, decimals=6),
        )
        frames.append(
            SimulationFrame(
                time_s=round(t, 6),
                step_index=index,
                plume_centroid_m=[round(x_c, 6), round(y_c, 6)],
                diffusion_radius_m=round(r_diff, 7),
                net_buoyancy_force_n=round(net_buoyancy_force, 12),
                spatial_grid=grid,
            )
        )
        time_series.append(
            SimulationPoint(
                time_s=round(t, 6),
                diffusion_length_scale_m=round(
                    math.sqrt(2.0 * request.diffusion_coefficient_m2_s * t), 8
                ),
                buoyancy_force_n=round(buoyancy_force, 12),
                hot_gas_weight_n=round(hot_weight, 12),
                net_buoyancy_force_n=round(net_buoyancy_force, 12),
                reduced_gravity_m_s2=round(reduced_gravity, 8),
                buoyancy_velocity_scale_m_s=round(buoyancy_velocity_scale, 8),
                forced_flow_speed_m_s=request.forced_flow_speed_m_s,
            )
        )

    equations_by_id = {item["id"]: item for item in config["equations"]}
    equation_ids = [
        "ideal_gas_density",
        "buoyancy_force",
        "hot_gas_weight",
        "net_buoyancy_force",
        "reduced_gravity",
        "buoyancy_velocity_scale",
        "diffusion_length_scale",
        "illustrative_hot_gas_centroid_motion",
        "illustrative_2d_gaussian_temperature_field",
    ]
    inputs = request.model_dump()
    inputs["steps"] = n_steps
    inputs.pop("fps", None)

    return SimulationResponse(
        model_id=config["model_id"],
        model_version=config["model_version"],
        model_type=config["model_type"],
        scientific_status="schematic_educational_approximation_not_validated_for_prediction",
        inputs=inputs,
        derived_values={
            "ambient_gas_density_kg_m3": rho_ambient,
            "hot_gas_density_kg_m3": rho_hot,
            "buoyancy_force_n": buoyancy_force,
            "hot_gas_weight_n": hot_weight,
            "net_buoyancy_force_n": net_buoyancy_force,
            "reduced_gravity_m_s2": reduced_gravity,
            "buoyancy_velocity_scale_m_s": buoyancy_velocity_scale,
            "forced_flow_speed_m_s": request.forced_flow_speed_m_s,
            "buoyancy_to_flow_scale_ratio": buoyancy_to_flow_ratio,
            "diffusion_time_scale_s": diffusion_time_scale,
            "buoyancy_time_scale_s": buoyancy_time_scale,
        },
        time_series=time_series,
        frames=frames,
        grid_metadata={
            "x_min_m": float(x_coords[0]),
            "x_max_m": float(x_coords[-1]),
            "y_min_m": float(y_coords[0]),
            "y_max_m": float(y_coords[-1]),
            "nx": request.grid_resolution,
            "ny": request.grid_resolution,
            "number_of_frames": len(frames),
            "field_status": "schematic_educational_field_not_empirical_data",
        },
        equations=[equations_by_id[item] for item in equation_ids if item in equations_by_id],
        assumptions=config.get("model_assumptions", []),
        limitations=config["science_notes"],
        references=config["reference_urls"],
    )
