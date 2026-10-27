"""Evaluation pipeline behaviour, generated report consistency, checkpoint, API and frontend status handling."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import provenance
from src import run_cv_evaluation as cv

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CLIENT = TestClient(app)
PROTOCOL = {"seeds": [1, 2, 3, 4, 5], "outer_folds": 5, "inner_validation_fraction": 0.2, "min_independent_groups": 5}


def _toy_task(task_type="regression"):
    task = {"task_id": "toy", "task_type": task_type, "record_id_column": "rid", "group_column": "g",
            "features": [{"column": "a", "available": "before_test"}, {"column": "b", "available": "before_test"}]}
    if task_type == "regression":
        task["target"] = {"column": "y", "unit": "u", "unit_verified": True}
    else:
        task["label"] = {"column": "y", "source_column": "y", "mapping_verified": True}
    return task


def _toy_frame(n_groups=12, per_group=4, classification=False, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n_groups * per_group), rng.normal(size=n_groups * per_group)
    y = 3 * a - b + rng.normal(scale=0.1, size=a.size) + 10
    if classification:
        y = (a + b > 0).astype(float)
    return pd.DataFrame({"rid": [f"r{i}" for i in range(a.size)], "g": np.repeat([f"g{i}" for i in range(n_groups)], per_group),
                         "a": a, "b": b, "y": y})


def test_preprocessing_is_fitted_on_training_rows_only():
    df = _toy_frame()
    x, y = df[["a", "b"]].to_numpy(), df["y"].to_numpy()
    train, test = np.arange(0, 32), np.arange(32, 48)
    base = cv.fit_predict("ridge", {"alpha": 1.0}, x[train], y[train], x[test], seed=1, task_type="regression")
    x_poisoned = x.copy()
    x_poisoned[test[1:]] *= 1000.0  # if scaling were fitted on test rows, row test[0] would change
    again = cv.fit_predict("ridge", {"alpha": 1.0}, x[train], y[train], x_poisoned[test], seed=1, task_type="regression")
    assert np.isclose(base[0], again[0])


def test_evaluate_task_regression_outputs_and_audit():
    task = _toy_task()
    df = _toy_frame()
    df.loc[0, "y"] = np.nan   # invalid targets are dropped before splitting, and counted
    df.loc[1, "y"] = -5.0
    summary, folds, manifest, preds = cv.evaluate_task(task, df, PROTOCOL, ["mean_baseline", "ridge"])
    assert summary["split_audit"]["passed"] and all(v == 0 for v in summary["split_audit"]["checks"].values())
    assert summary["rows_excluded_before_split"] == 2
    assert len(folds) == 25 * 2
    assert set(folds[0]) <= set(cv.FOLD_COLUMNS)
    assert len(preds) == (len(df) - 2) * 5 * 2  # every usable row predicted once per repeat per model
    assert summary["models"]["ridge"]["fold_level"]["mae"]["mean"] < summary["models"]["mean_baseline"]["fold_level"]["mae"]["mean"]


def test_classification_metrics_refuse_single_class_folds():
    m = cv.classification_metrics(np.array([1, 1, 1]), np.array([0.9, 0.2, 0.8]), 0.5)
    assert m["balanced_accuracy"] is None and m["roc_auc"] is None and m["metric_note"]
    m2 = cv.classification_metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.8, 0.4]), 0.5)
    assert m2["balanced_accuracy"] == 1.0 and m2["roc_auc"] == 1.0


def test_eligibility_blocks_unverified_data_units_and_labels():
    verified = {"datasets": {"X": {"status": "VERIFIED_SOURCE"}}}
    synthetic = {"datasets": {"X": {"status": "SYNTHETIC_DEMO"}}}
    task = {**_toy_task(), "dataset_key": "X"}
    assert [r["code"] for r in cv.eligibility(task, synthetic, 5)] == ["DATASET_NOT_VERIFIED"]
    unverified_unit = {**task, "target": {"column": "y", "unit_verified": False}}
    assert "TARGET_UNIT_UNVERIFIED" in [r["code"] for r in cv.eligibility(unverified_unit, verified, 5)]
    label_task = {**_toy_task("classification"), "dataset_key": "X", "label": {"column": "y", "source_column": "y", "mapping_verified": False}}
    assert "LABEL_MAPPING_UNVERIFIED" in [r["code"] for r in cv.eligibility(label_task, verified, 5)]


def _recheck(man):
    issues = 0
    for _, s in man.groupby(["task_id", "seed", "fold"]):
        p = {k: set(s.loc[s["partition"] == k, "group_id"]) for k in ("inner_train", "inner_validation", "outer_test")}
        issues += len(p["inner_train"] & p["inner_validation"]) + len(p["inner_train"] & p["outer_test"]) + len(p["inner_validation"] & p["outer_test"])
        issues += int(s["record_id"].duplicated().sum())
    tested = man[man["partition"] == "outer_test"].drop_duplicates(["task_id", "seed", "fold", "group_id"])
    issues += int((tested.groupby(["task_id", "seed", "group_id"]).size() != 1).sum())
    return issues


def test_generated_reports_are_consistent_and_leakage_free():
    summary = json.loads((REPORTS / "cv_summary.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(REPORTS / "cv_fold_metrics.csv")
    manifest = pd.read_csv(REPORTS / "experiment_split_manifest.csv")
    registry = json.loads((ROOT / "config" / "target_registry.json").read_text(encoding="utf-8"))
    assert {t["task_id"] for t in summary["tasks"]} == {t["task_id"] for t in registry["tasks"]}
    assert _recheck(manifest) == 0
    for t in summary["tasks"]:
        rows = folds[folds["task_id"] == t["task_id"]]
        if t["evaluation_status"] == "EVALUATED":
            assert t["split_audit"]["passed"]
            assert len(rows) == 25 * len(t["models"])
            assert (rows.groupby("model").size() == 25).all()
        else:
            assert t["reasons"] and rows.empty  # no scores for tasks that were not evaluated
    protocol = json.loads((REPORTS / "cv_protocol.json").read_text(encoding="utf-8"))
    assert len(protocol["seeds"]) == 5 and protocol["outer_folds_per_seed"] == 5


def test_deployed_checkpoint_matches_its_metadata_and_training_file():
    meta = json.loads((ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json").read_text())
    data = ROOT / "data" / "processed" / "macfp_measured_cone_experiments.csv"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == meta["training_data"]["csv_sha256"]
    from app.services.model_inference import FEATURE_COLUMNS
    assert meta["feature_columns"] == list(FEATURE_COLUMNS)
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.models.load_model(ROOT / "models" / "spacecraft_fire_peak_hrr_v1.keras", compile=False, safe_mode=True)
    assert model.input_shape == (None, 3) and model.output_shape == (None, 1)
    out = model(np.zeros((1, 3), dtype=np.float32), training=False).numpy()
    assert np.isfinite(out).all() and (out >= 0).all()


def test_dataset_status_is_explicit_when_report_missing_or_synthetic(tmp_path):
    missing = provenance.dataset_status("FLEX", path=tmp_path / "missing.json")
    assert missing["status"] == "UNVERIFIED_SOURCE" and "missing" in missing["warning"]
    synthetic = provenance.dataset_status("FLEX", {"datasets": {"FLEX": {"status": "SYNTHETIC_DEMO"}}})
    assert synthetic["status"] == "SYNTHETIC_DEMO" and "synthetic" in synthetic["warning"]
    verified = provenance.dataset_status("FLEX", {"datasets": {"FLEX": {"status": "VERIFIED_SOURCE"}}})
    assert verified["warning"] is None


def test_api_reports_statuses_and_result_types(tmp_path, monkeypatch):
    body = CLIENT.get("/api/v1/research/provenance").json()
    assert body["report_available"] is True
    assert {d["status"] for d in body["datasets"]} <= {"VERIFIED_SOURCE", "UNVERIFIED_SOURCE", "SYNTHETIC_DEMO"}
    sweep = CLIENT.post("/api/v1/research/parametric-sweep", json={}).json()
    assert sweep["result_type"] == "SIMULATED_RESULT"
    fake = tmp_path / "report.json"
    fake.write_text(json.dumps({"datasets": {"BASS_II": {"status": "SYNTHETIC_DEMO"}}}))
    monkeypatch.setattr(provenance, "REPORT_PATH", fake)
    ranking = CLIENT.get("/api/v1/research/empirical-ranking?track_key=solid_fuel").json()
    assert ranking["data_status"] == "SYNTHETIC_DEMO" and ranking["warning"]
    monkeypatch.setattr(provenance, "REPORT_PATH", tmp_path / "absent.json")
    unavailable = CLIENT.get("/api/v1/research/provenance").json()
    assert unavailable["report_available"] is False
    assert all(d["status"] == "UNVERIFIED_SOURCE" and d["warning"] for d in unavailable["datasets"])


def test_predict_fire_is_labelled_as_model_prediction(monkeypatch):
    from app.services import model_inference

    class FakeModel:
        def __call__(self, features, training=False):
            return features[:, :1] * 0 + 500

    metadata = {"model_id": "m", "source": "s", "scope": "post-test", "feature_columns": list(model_inference.FEATURE_COLUMNS),
                "scaler": {"mean": [0, 0, 0], "scale": [1, 1, 1]}, "hazard_threshold_kw_m2": 350.0, "validation": {},
                "training_data": {"feature_ranges": {c: {"min": 0, "max": 5000} for c in model_inference.FEATURE_COLUMNS}}}
    monkeypatch.setattr(model_inference, "_get_bundle", lambda: (FakeModel(), metadata))
    row = {"radiant_heat_flux_kw_m2": 50, "real_burn_duration_s": 400, "real_mean_hrr_kw_m2": 500}
    body = CLIENT.post("/api/v1/research/predict-fire", json={"rows": [row]}).json()
    assert body["result_type"] == "MODEL_PREDICTION" and body["training_data_status"]


def test_dashboard_shows_an_error_instead_of_silently_trusting_data(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(provenance, "REPORT_PATH", tmp_path / "absent.json")
    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=240).run()
    assert not at.exception
    assert any("Provenance report missing" in e.value for e in at.error)
    st.cache_data.clear()
