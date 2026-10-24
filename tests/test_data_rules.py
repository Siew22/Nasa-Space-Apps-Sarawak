"""Parsing rules for the official NASA PSI tables and the FLEX task-table builder."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.psi_parsing import (CENSORED_ABOVE, MISSING, MULTIPLE_VALUES, NO_VALUE, OK, OVER_RANGE,
                                      APPROXIMATE, bass_material_class, numeric_cell)

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "processed" / "dashboard_records.csv"
TASKS = ROOT / "data" / "processed" / "tasks" / "flex_tests.csv"


@pytest.mark.parametrize("cell, value, flag", [
    ("12", 12.0, OK), (7, 7.0, OK), ("OR", None, OVER_RANGE), ("over", None, OVER_RANGE),
    (">500", None, CENSORED_ABOVE), ("–", None, NO_VALUE), ("none", None, MISSING), (float("nan"), None, MISSING),
    ("~6", 6.0, APPROXIMATE), ("0.9   1.3   2.75", None, MULTIPLE_VALUES),
])
def test_numeric_cell_classifies_instead_of_guessing(cell, value, flag):
    assert numeric_cell(cell) == (value, flag)


@pytest.mark.parametrize("text, cls", [
    ('1/4" clear rod', "pmma_rod_clear"), ("1/4 blac rod", "pmma_rod_black"), ("2 cm Sibal   moved igniter", "sibal_fabric"),
    ("2 cm   0.1 mm PMMA film", "pmma_thin_film"), ("1 cm wide 200 micron thick PMMA", "pmma_thin_film"),
    ("4 mm thick   2 cm 2ide PMMA slab", "pmma_slab"), ("2 cm wide PMMA 3 mm thick", "pmma_slab"),
    ("Nomex natural (nomex III)", "nomex_fabric"), ("4 cm spherical section", "pmma_sphere_section"),
    ("Japan Wax candle", "wax_candle"),
])
def test_bass_material_classes(text, cls):
    assert bass_material_class(text) == cls


def test_every_real_bass_record_gets_a_material_class():
    records = pd.read_csv(RECORDS, low_memory=False)
    bass = records[records["dataset_key"] == "BASS_II"]
    assert len(bass) > 0
    assert (bass["fuel_sample_material"].map(bass_material_class) != "other").all()


def test_flex_task_table_schema_exclusions_and_labels():
    tasks = pd.read_csv(TASKS)
    expected = {"record_id", "flex_identifier", "group_test_date", "fuel", "is_heptane", "pressure_mmhg", "x_o2", "x_n2",
                "x_co2", "x_he", "d0_mm", "burning_rate_constant_mm2_s", "test_end", "extinction_label",
                "use_burning_rate_task", "use_extinction_task"}
    assert expected <= set(tasks.columns)
    assert tasks["record_id"].is_unique
    # Conflicting duplicate identifier is excluded from both tasks, not silently de-duplicated.
    dup = tasks["flex_identifier"].duplicated(keep=False)
    assert not tasks.loc[dup, ["use_burning_rate_task", "use_extinction_task"]].any().any()
    # A physically invalid pressure never enters a task.
    assert not tasks.loc[tasks["pressure_mmhg"] < 1.0, ["use_burning_rate_task", "use_extinction_task"]].any().any()
    # Label rule: Extinction = 1, Completion = 0, Disruption is excluded (not pooled into the negative class).
    lab = tasks.dropna(subset=["extinction_label"])
    assert set(lab.loc[lab["test_end"] == "Extinction", "extinction_label"]) == {1.0}
    assert set(lab.loc[lab["test_end"] == "Completion", "extinction_label"]) == {0.0}
    disrupted = tasks[tasks["test_end"] == "Disruption"]
    assert disrupted["extinction_label"].isna().all() and not disrupted["use_extinction_task"].any()
    # Every row used for regression has a finite positive target and complete inputs.
    reg = tasks[tasks["use_burning_rate_task"]]
    assert np.isfinite(reg["burning_rate_constant_mm2_s"]).all() and (reg["burning_rate_constant_mm2_s"] > 0).all()
    assert reg[["pressure_mmhg", "x_o2", "x_co2", "x_he", "d0_mm"]].notna().all().all()


def test_every_flex_exclusion_has_a_reason():
    excl = pd.read_csv(TASKS.with_name("flex_exclusions.csv"))
    assert excl["reason"].str.len().gt(10).all()
    meta = json.loads(TASKS.with_name("flex_tests.metadata.json").read_text())
    assert meta["rows"] == len(pd.read_csv(TASKS))
    assert "mm^2/s" in meta["columns"]["burning_rate_constant_mm2_s"]
