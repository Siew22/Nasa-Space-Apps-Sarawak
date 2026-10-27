"""Focused API checks for optional saved-model inference."""

from fastapi.testclient import TestClient

from app.main import app
from app.services import model_inference


CLIENT = TestClient(app)
ENDPOINT = "/api/v1/research/predict-fire"
ROW = {
    "radiant_heat_flux_kw_m2": 50,
    "real_burn_duration_s": 450,
    "real_mean_hrr_kw_m2": 320,
}


def test_predict_fire_requires_saved_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(model_inference, "MODEL_PATH", tmp_path / "missing.keras")
    monkeypatch.setattr(model_inference, "METADATA_PATH", tmp_path / "missing.metadata.json")

    response = CLIENT.post(ENDPOINT, json={"rows": [ROW]})

    assert response.status_code == 503
    assert "trained prediction model" in response.json()["detail"]


def test_predict_fire_rejects_invalid_numeric_input():
    response = CLIENT.post(ENDPOINT, json={"rows": [{**ROW, "real_mean_hrr_kw_m2": "unknown"}]})

    assert response.status_code == 422
    zero_duration = CLIENT.post(ENDPOINT, json={"rows": [{**ROW, "real_burn_duration_s": 0}]})
    assert zero_duration.status_code == 422


def test_predict_fire_uses_saved_scaler_and_project_threshold(monkeypatch):
    class FakeModel:
        def __call__(self, features, training=False):
            assert training is False
            return features[:, :1] * 100 + 300

    metadata = {
        "model_id": "test-model",
        "source": "measured test rows",
        "scope": "post-test burn summaries",
        "feature_columns": list(model_inference.FEATURE_COLUMNS),
        "scaler": {"mean": [50, 0, 0], "scale": [50, 1, 1]},
        "hazard_threshold_kw_m2": 350.0,
        "validation": {"hazard_discrimination_validated": False},
        "training_data": {
            "feature_ranges": {
                "radiant_heat_flux_kw_m2": {"min": 50, "max": 100},
                "real_burn_duration_s": {"min": 400, "max": 500},
                "real_mean_hrr_kw_m2": {"min": 300, "max": 400},
            }
        },
    }
    monkeypatch.setattr(model_inference, "_get_bundle", lambda: (FakeModel(), metadata))

    second_row = {**ROW, "radiant_heat_flux_kw_m2": 100}
    response = CLIENT.post(ENDPOINT, json={"rows": [ROW, second_row]})

    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "measured test rows"
    assert result["validation"]["hazard_discrimination_validated"] is False
    assert [item["predicted_peak_hrr_kw_m2"] for item in result["predictions"]] == [300, 400]
    assert [item["predicted_high_hazard"] for item in result["predictions"]] == [False, True]

    outside_range = CLIENT.post(ENDPOINT, json={"rows": [{**ROW, "radiant_heat_flux_kw_m2": 101}]})
    assert outside_range.status_code == 422
    assert "outside the measured training range" in outside_range.json()["detail"]
