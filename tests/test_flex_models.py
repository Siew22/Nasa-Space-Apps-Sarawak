"""The deployed FLEX models: exact export, reproducible training, honest metadata and safe inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.services import flex_inference as flex
from src import train_flex_models as trainer
from src.run_cv_evaluation import make_model

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {t["task_id"]: t for t in json.loads((ROOT / "config" / "target_registry.json").read_text(encoding="utf-8"))["tasks"]}
CV = {t["task_id"]: t for t in json.loads((ROOT / "reports" / "cv_summary.json").read_text(encoding="utf-8"))["tasks"]}
SEEDS = json.loads((ROOT / "reports" / "cv_protocol.json").read_text(encoding="utf-8"))["seeds"]
FILES = {task_id: ROOT / "models" / f for task_id, (_, _, f) in trainer.DEPLOY.items()}
AIR = dict(is_heptane=1.0, pressure_mmhg=760.0, x_o2=0.21, x_co2=0.0, x_he=0.0, d0_mm=3.0)


@pytest.mark.parametrize("task_id", list(trainer.DEPLOY))
def test_exported_numbers_reproduce_the_sklearn_pipeline(task_id):
    meta = flex.load_model(FILES[task_id])
    _, x, y, _ = trainer.task_rows(REGISTRY[task_id])
    pipe = make_model(meta["model_family"], meta["hyperparameters"], SEEDS[0]).fit(x, y)
    ours = np.array([flex._linear(meta, row)[0] for row in x])
    if meta["task_type"] == "regression":
        np.testing.assert_allclose(ours, pipe.predict(x), rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_allclose(1 / (1 + np.exp(-ours)), pipe.predict_proba(x)[:, 1], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("task_id", list(trainer.DEPLOY))
def test_retraining_gives_the_committed_model(task_id):
    family, baseline, _ = trainer.DEPLOY[task_id]
    cv_summary = json.loads((ROOT / "reports" / "cv_summary.json").read_text(encoding="utf-8"))
    fresh = trainer.export(REGISTRY[task_id], family, baseline, SEEDS, cv_summary)
    saved = flex.load_model(FILES[task_id])
    assert fresh["hyperparameters"] == saved["hyperparameters"]
    for key in ("coef", "intercept", "imputer_median", "scaler_mean", "scaler_scale"):
        np.testing.assert_allclose(fresh[key], saved[key], rtol=1e-10)
    chosen = min(saved["hyperparameter_selection"]["candidates"], key=lambda c: c["mean_score"])
    assert chosen["params"] == saved["hyperparameters"]


@pytest.mark.parametrize("task_id", list(trainer.DEPLOY))
def test_model_file_matches_its_data_and_the_cv_report(task_id):
    meta = flex.load_model(FILES[task_id])
    data = ROOT / meta["training_data"]["file"]
    assert meta["training_data"]["csv_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert meta["training_data"]["rows"] == CV[task_id]["rows_used"]
    assert meta["training_data"]["independent_groups"] == CV[task_id]["independent_groups"]
    assert meta["cross_validation"]["model"] == CV[task_id]["models"][meta["model_family"]]["fold_level"]
    assert meta["cross_validation"]["baseline"] == CV[task_id]["models"][meta["cross_validation"]["baseline_family"]]["fold_level"]


def test_contributions_add_up_to_the_prediction():
    p = flex.predict(AIR)
    assert p["burning_rate_at_average_test"] + sum(p["burning_rate_contributions"].values()) == pytest.approx(p["burning_rate_mm2_s"])
    assert p["extinction_logit_at_average_test"] + sum(p["extinction_contributions"].values()) == pytest.approx(p["extinction_logit"])
    assert p["burnout_time_s"] == pytest.approx(AIR["d0_mm"] ** 2 / p["burning_rate_mm2_s"])
    assert p["flame_goes_out"] == (p["extinction_score"] >= p["extinction_threshold"])


def test_more_oxygen_burns_faster_and_more_co2_slower():
    base = flex.predict(AIR)["burning_rate_mm2_s"]
    assert flex.predict({**AIR, "x_o2": 0.30})["burning_rate_mm2_s"] > base
    assert flex.predict({**AIR, "x_co2": 0.40})["burning_rate_mm2_s"] < base


@pytest.mark.parametrize("bad, why", [
    ({"pressure_mmhg": 5000.0}, "outside the range"),
    ({"x_o2": 0.05}, "outside the range"),
    ({"x_o2": 0.34, "x_co2": 0.70}, "add up to more than 1"),
    ({"is_heptane": 0.5}, "is_heptane"),
    ({"d0_mm": True}, "finite number"),
    ({"d0_mm": float("nan")}, "finite number"),
])
def test_unsafe_inputs_are_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        flex.predict({**AIR, **bad})


def test_nearest_tests_are_real_rows_with_the_same_fuel_and_close_conditions():
    near = flex.nearest_tests(AIR)
    df = flex._tests(str(flex.TESTS_PATH), flex.TESTS_PATH.stat().st_mtime_ns)
    manual = df[(df.is_heptane == 1) & ((df.pressure_mmhg - 760).abs() <= 76) & ((df.x_o2 - 0.21).abs() <= 0.02 + 1e-9)
                & (df.x_co2 <= 0.05 + 1e-9) & (df.x_he <= 0.05 + 1e-9)]
    assert near["tests"] == len(manual) > 0
    assert near["went_out"] + near["burned_out"] + near["disrupted"] == near["tests"]


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(flex.FlexModelUnavailable, match="train_flex_models"):
        flex.load_model(ROOT / "models" / "does_not_exist.json")
