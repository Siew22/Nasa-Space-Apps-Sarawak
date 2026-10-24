"""Provenance statuses are computed from evidence; tasks never merge physically different targets."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from src import fetch_nasa_psi_tables, provenance_audit
from src.build_dashboard_data import normalize_column_name

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / "config" / "target_registry.json").read_text(encoding="utf-8"))
SOURCES = json.loads((ROOT / "config" / "data_sources.json").read_text(encoding="utf-8"))
VOCAB = {"VERIFIED_SOURCE", "UNVERIFIED_SOURCE", "SYNTHETIC_DEMO"}
EMPTY_CATALOG = pd.DataFrame(columns=["dataset_key", "original_field_name", "field_name"])


def test_official_nasa_files_match_registered_checksums():
    entries = fetch_nasa_psi_tables.official_files()
    assert {e["dataset_key"] for e in entries} == {"BASS_II", "FLEX"}
    for entry in entries:
        ok, why = fetch_nasa_psi_tables.verify_local(entry)
        assert ok, why


def test_generated_rows_are_classified_synthetic_not_verified():
    from src import generate_synthetic_demo_tables as gen

    with tempfile.TemporaryDirectory() as tmp:
        df = pd.read_csv(gen.generate_flex_dataset(Path(tmp)))
    df.columns = [normalize_column_name(c) for c in df.columns]
    df["dataset_key"], df["source_row_number"] = "FLEX", range(1, len(df) + 1)
    df["record_id"] = [f"FLEX:demo:{i}" for i in df["source_row_number"]]
    df["source_file"] = "data/demo/SYNTHETIC_DEMO_FLEX_PSI-69_format.csv"
    summary, mapping = provenance_audit.audit_nasa_format_dataset("FLEX", df, SOURCES, EMPTY_CATALOG)
    assert summary["status"] == "SYNTHETIC_DEMO"
    assert summary["independent_experiments_verified"] == 0
    assert {m["verification_status"] for m in mapping} == {"SYNTHETIC_DEMO"}


def test_unregistered_real_looking_data_is_unverified():
    records = pd.read_csv(ROOT / "data" / "processed" / "dashboard_records.csv", low_memory=False)
    sources = json.loads(json.dumps(SOURCES))
    sources["datasets"]["FLEX"]["official_files"] = []  # no checksum on record -> cannot be verified
    summary, _ = provenance_audit.audit_nasa_format_dataset("FLEX", records, sources, EMPTY_CATALOG)
    assert summary["status"] == "UNVERIFIED_SOURCE"
    assert summary["required_to_verify"]


def test_generated_provenance_report_and_mapping_are_consistent():
    report = json.loads((ROOT / "reports" / "data_provenance_audit.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(ROOT / "reports" / "provenance_mapping.csv")
    records = pd.read_csv(ROOT / "data" / "processed" / "dashboard_records.csv", low_memory=False)
    for key in ("BASS_II", "FLEX"):
        entry = report["datasets"][key]
        assert entry["status"] == "VERIFIED_SOURCE"
        assert "OFFICIAL_CHECKSUM_MATCH" in [e["code"] for e in entry["evidence"]]
        assert entry["rows"] == int((records["dataset_key"] == key).sum())
    assert report["datasets"]["MACFP_PMMA_CONE"]["status"] in VOCAB
    assert set(mapping["verification_status"]) <= VOCAB
    assert mapping["internal_record_id"].is_unique
    assert len(mapping) == len(records) + report["datasets"]["MACFP_PMMA_CONE"]["rows"]


def test_registry_never_merges_targets_and_units_are_declared():
    seen = {}
    for task in REGISTRY["tasks"]:
        column = (task.get("target") or {}).get("column") if task["task_type"] == "regression" else (task.get("label") or {}).get("column")
        if column:
            key = (task.get("data_file"), column)
            assert key not in seen or seen[key] == task["dataset_key"], "one target column used by two datasets"
            seen[key] = task["dataset_key"]
        if task["task_type"] == "regression" and (task.get("target") or {}).get("unit_verified"):
            assert task["target"]["unit"] and task["target"].get("unit_evidence")
        if task.get("label", {}).get("mapping_verified"):
            mapping = task["label"]["mapping"]
            assert "excluded" in mapping.get("Disruption", "")  # never pooled into the negative class
    units = {t["task_id"]: (t.get("target") or {}).get("unit") for t in REGISTRY["tasks"]}
    assert units["macfp_pmma_peak_hrr"] == "kW/m2"
    assert units["flex_droplet_burning_rate_constant"] == "mm^2/s"
    spread = next(t for t in REGISTRY["tasks"] if t["task_id"] == "bass_ii_flame_spread_speed")
    assert spread["target"]["column"] is None  # no measured spread speed in the PSI-25 table


def test_model_features_are_known_before_the_test_where_claimed():
    for task in REGISTRY["tasks"]:
        for feature in task.get("features", []):
            assert feature["available"] in {"before_test", "after_test"}
        if task["dataset_key"] == "FLEX":
            assert all(f["available"] == "before_test" for f in task["features"])
