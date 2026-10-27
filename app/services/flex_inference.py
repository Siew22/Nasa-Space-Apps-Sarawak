"""Predictions from the two deployed FLEX models (microgravity droplet flames, ISS).

The model files (models/flex_*_v1.json, written by src/train_flex_models.py) hold the fitted imputation medians,
scaling and coefficients, so inference is plain numpy: no pickle, no scikit-learn. Both models are linear, so each
prediction splits exactly into one contribution per input. Inputs outside the training ranges are refused rather
than extrapolated, and the closest real NASA tests are returned alongside every prediction.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BURNING_RATE_PATH = ROOT / "models" / "flex_burning_rate_ridge_v1.json"
EXTINCTION_PATH = ROOT / "models" / "flex_extinction_logistic_v1.json"
TESTS_PATH = ROOT / "data" / "processed" / "tasks" / "flex_tests.csv"
FEATURES = ("is_heptane", "pressure_mmhg", "x_o2", "x_co2", "x_he", "d0_mm")
# "Close" NASA test: same fuel, pressure within 10 %, each gas mole fraction within these absolute margins.
NEAR_PRESSURE_REL, NEAR_O2, NEAR_DILUENT = 0.10, 0.02, 0.05


class FlexModelUnavailable(RuntimeError):
    """A deployed FLEX model file is missing or does not match the expected inputs."""


@lru_cache(maxsize=4)
def _read(path_str: str, mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FlexModelUnavailable(f"{path.relative_to(ROOT)} is missing. Run `python -m src.train_flex_models`.")
    model = _read(str(path), path.stat().st_mtime_ns)
    if tuple(model.get("feature_columns", ())) != FEATURES:
        raise FlexModelUnavailable(f"{path.name} expects inputs {model.get('feature_columns')}, not {list(FEATURES)}.")
    return model


def input_ranges() -> dict[str, tuple[float, float]]:
    """Ranges both models were trained on (the intersection, so neither model extrapolates)."""
    ranges = [load_model(p)["training_data"]["feature_ranges"] for p in (BURNING_RATE_PATH, EXTINCTION_PATH)]
    return {c: (max(r[c]["min"] for r in ranges), min(r[c]["max"] for r in ranges)) for c in FEATURES}


def _vector(inputs: dict[str, Any]) -> np.ndarray:
    values = []
    for c in FEATURES:
        v = inputs.get(c)
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
            raise ValueError(f"{c} must be a finite number")
        values.append(float(v))
    x = np.array(values)
    if x[0] not in (0.0, 1.0):
        raise ValueError("is_heptane must be 1 (n-heptane) or 0 (methanol)")
    if x[2] + x[3] + x[4] > 1.0 + 1e-9:
        raise ValueError("O2, CO2 and He mole fractions add up to more than 1")
    for c, v in zip(FEATURES, x):
        lo, hi = input_ranges()[c]
        if not lo - 1e-9 <= v <= hi + 1e-9:
            raise ValueError(f"{c} = {v:g} is outside the range NASA tested ({lo:g} to {hi:g}); the model will not extrapolate")
    return x


def _linear(model: dict[str, Any], x: np.ndarray) -> tuple[float, np.ndarray]:
    """Intercept + per-input contributions, exactly as the exported pipeline computes them."""
    filled = np.where(np.isnan(x), model["imputer_median"], x)
    z = (filled - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    contrib = np.array(model["coef"]) * z
    return float(model["intercept"] + contrib.sum()), contrib


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    """Burning-rate constant K and the flame-out call for one droplet test condition."""
    x = _vector(inputs)
    k_model, e_model = load_model(BURNING_RATE_PATH), load_model(EXTINCTION_PATH)
    k, k_contrib = _linear(k_model, x)
    if k <= 0:
        raise ValueError("this combination is outside what the burning-rate model can describe (predicted K <= 0)")
    logit, e_contrib = _linear(e_model, x)
    score = 1.0 / (1.0 + math.exp(-logit))
    threshold = float(e_model.get("decision_threshold") or 0.5)
    return {
        "result_type": "MODEL_PREDICTION",
        "burning_rate_mm2_s": k,
        "burning_rate_at_average_test": float(k_model["intercept"]),
        "burning_rate_contributions": dict(zip(FEATURES, k_contrib.tolist())),
        # d^2-law: d(t)^2 = d0^2 - K t, so a droplet that keeps burning is used up after d0^2 / K.
        "burnout_time_s": x[5] ** 2 / k,
        "extinction_score": score,
        "extinction_threshold": threshold,
        "flame_goes_out": score >= threshold,
        "extinction_logit": logit,
        "extinction_logit_at_average_test": float(e_model["intercept"]),
        "extinction_contributions": dict(zip(FEATURES, e_contrib.tolist())),
    }


@lru_cache(maxsize=2)
def _tests(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_str)


def nearest_tests(inputs: dict[str, Any]) -> dict[str, Any]:
    """Real NASA FLEX tests run under a similar condition (same fuel, close pressure and gas mix), any droplet size."""
    x = _vector(inputs)
    if not TESTS_PATH.exists():
        return {"tests": 0}
    df = _tests(str(TESTS_PATH), TESTS_PATH.stat().st_mtime_ns)
    near = df[(df["is_heptane"] == x[0])
              & ((df["pressure_mmhg"] - x[1]).abs() <= NEAR_PRESSURE_REL * x[1])
              & ((df["x_o2"] - x[2]).abs() <= NEAR_O2 + 1e-9)
              & ((df["x_co2"] - x[3]).abs() <= NEAR_DILUENT + 1e-9)
              & ((df["x_he"] - x[4]).abs() <= NEAR_DILUENT + 1e-9)]
    k = pd.to_numeric(near["burning_rate_constant_mm2_s"], errors="coerce").dropna()
    k = k[k > 0]
    ends = near["test_end"].astype(str).str.strip().value_counts()
    return {
        "tests": int(len(near)),
        "k_readings": int(len(k)),
        "k_mean": float(k.mean()) if len(k) else None,
        "k_min": float(k.min()) if len(k) else None,
        "k_max": float(k.max()) if len(k) else None,
        "went_out": int(ends.get("Extinction", 0)),
        "burned_out": int(ends.get("Completion", 0)),
        "disrupted": int(ends.get("Disruption", 0)),
        "d0_min": float(near["d0_mm"].min()) if len(near) else None,
        "d0_max": float(near["d0_mm"].max()) if len(near) else None,
    }
