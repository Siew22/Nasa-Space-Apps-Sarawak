"""Post-test peak heat-release predictions from an explicitly saved model.

TensorFlow is imported only when inference is requested. A deployment without
trained artifacts can still start and serve the other research endpoints.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "spacecraft_fire_peak_hrr_v1.keras"
METADATA_PATH = ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json"
FEATURE_COLUMNS = (
    "radiant_heat_flux_kw_m2",
    "real_burn_duration_s",
    "real_mean_hrr_kw_m2",
)
MAX_BATCH_ROWS = 1000
_PREDICTION_LOCK = Lock()


class ModelUnavailableError(RuntimeError):
    """The trained artifact needed for prediction is absent or unusable."""


def _artifact_stamp(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelUnavailableError(
            "No trained prediction model is available. Deploy both "
            "models/spacecraft_fire_peak_hrr_v1.keras and "
            "models/spacecraft_fire_peak_hrr_v1.metadata.json to enable this endpoint."
        ) from exc
    return stat.st_mtime_ns, stat.st_size


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        requirement = "a positive finite number" if positive else "a finite number"
        raise ValueError(f"{label} must be {requirement}")
    return result


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError("feature_columns do not match the API input order")
        scaler = metadata.get("scaler")
        if not isinstance(scaler, dict):
            raise ValueError("scaler is missing")
        for name, positive in (("mean", False), ("scale", True)):
            values = scaler.get(name)
            if not isinstance(values, list) or len(values) != len(FEATURE_COLUMNS):
                raise ValueError(f"scaler.{name} must contain three numbers")
            for index, value in enumerate(values):
                _finite_number(value, f"scaler.{name}[{index}]", positive=positive)
        threshold = _finite_number(
            metadata.get("hazard_threshold_kw_m2"), "hazard_threshold_kw_m2", positive=True
        )
        if threshold != 350.0:
            raise ValueError("hazard threshold does not match the project definition")
        if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
            raise ValueError("source is missing")
        if not isinstance(metadata.get("scope"), str) or not metadata["scope"].strip():
            raise ValueError("scope is missing")
        training_data = metadata.get("training_data")
        ranges = training_data.get("feature_ranges") if isinstance(training_data, dict) else None
        if not isinstance(ranges, dict):
            raise ValueError("training_data.feature_ranges is missing")
        for column in FEATURE_COLUMNS:
            bounds = ranges.get(column)
            if not isinstance(bounds, dict):
                raise ValueError(f"training range for {column} is missing")
            lower = _finite_number(bounds.get("min"), f"{column} training minimum")
            upper = _finite_number(bounds.get("max"), f"{column} training maximum")
            if lower < 0 or upper < lower:
                raise ValueError(f"training range for {column} is invalid")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ModelUnavailableError(
            "Saved model metadata is missing or incompatible with this API. "
            "Recreate and deploy the matching model and metadata together."
        ) from exc
    return metadata


@lru_cache(maxsize=1)
def _load_bundle(
    model_path: str,
    model_stamp: tuple[int, int],
    metadata_path: str,
    metadata_stamp: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    del model_stamp, metadata_stamp  # Cache keys trigger reload when either file changes.
    metadata = _read_metadata(Path(metadata_path))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow import keras

        model = keras.models.load_model(model_path, compile=False, safe_mode=True)
        if model.input_shape != (None, len(FEATURE_COLUMNS)) or model.output_shape != (None, 1):
            raise ValueError("model input or output shape does not match the API contract")
    except Exception as exc:
        raise ModelUnavailableError(
            "Saved prediction model could not be loaded or does not match its metadata. "
            "Recreate and deploy the matching model and metadata together."
        ) from exc
    return model, metadata


def _get_bundle() -> tuple[Any, dict[str, Any]]:
    model_stamp = _artifact_stamp(MODEL_PATH)
    metadata_stamp = _artifact_stamp(METADATA_PATH)
    return _load_bundle(str(MODEL_PATH), model_stamp, str(METADATA_PATH), metadata_stamp)


def get_model_info() -> dict[str, Any]:
    """Return validated information about the deployed, loadable model."""
    _, metadata = _get_bundle()
    return {
        "model_id": metadata.get("model_id", MODEL_PATH.stem),
        "source": metadata["source"],
        "scope": metadata["scope"],
        "hazard_threshold_kw_m2": metadata["hazard_threshold_kw_m2"],
        "validation": metadata.get("validation"),
        "training_data": metadata.get("training_data"),
    }


def predict_fire_rows(rows: list[dict[str, float]]) -> list[dict[str, float | bool]]:
    """Predict peak HRR from measured post-test summaries; derive the hazard flag.

    This is a project-defined threshold on a regression result, not an
    independently trained classifier or a fire-safety determination.
    """
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_BATCH_ROWS:
        raise ValueError(f"rows must contain 1 to {MAX_BATCH_ROWS} records")
    feature_rows: list[list[float]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {row_number} must be an object")
        values: list[float] = []
        for column in FEATURE_COLUMNS:
            try:
                value = _finite_number(row[column], f"row {row_number} {column}")
            except KeyError as exc:
                raise ValueError(f"row {row_number} is missing {column}") from exc
            if column == FEATURE_COLUMNS[0] and value < 0:
                raise ValueError(f"row {row_number} {column} must be nonnegative")
            if column != FEATURE_COLUMNS[0] and value <= 0:
                raise ValueError(f"row {row_number} {column} must be positive")
            values.append(value)
        feature_rows.append(values)

    model, metadata = _get_bundle()
    ranges = metadata["training_data"]["feature_ranges"]
    for row_number, values in enumerate(feature_rows, start=1):
        for column, value in zip(FEATURE_COLUMNS, values):
            lower = float(ranges[column]["min"])
            upper = float(ranges[column]["max"])
            tolerance = max(1e-6, max(abs(lower), abs(upper)) * 1e-6)
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"row {row_number} {column}={value:g} is outside the measured training range "
                    f"[{lower:g}, {upper:g}]; use a model trained on matching conditions"
                )
    scaler = metadata["scaler"]
    data = np.asarray(feature_rows, dtype=np.float64)
    scaled = (data - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if not np.isfinite(scaled).all() or not np.isfinite(scaled.astype(np.float32)).all():
        raise ValueError("input values are outside the supported numeric range")
    try:
        with _PREDICTION_LOCK:
            raw_predictions = model(scaled.astype(np.float32), training=False)
        predictions = np.asarray(raw_predictions, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ModelUnavailableError("Saved prediction model failed during inference.") from exc
    if len(predictions) != len(rows) or not np.isfinite(predictions).all():
        raise ModelUnavailableError("Saved prediction model returned invalid output.")
    if (predictions < 0).any():
        raise ModelUnavailableError("Saved prediction model returned a negative heat-release estimate.")
    threshold = float(metadata["hazard_threshold_kw_m2"])
    return [
        {
            "predicted_peak_hrr_kw_m2": float(peak),
            "predicted_high_hazard": bool(peak >= threshold),
            "hazard_threshold_kw_m2": threshold,
        }
        for peak in predictions
    ]
