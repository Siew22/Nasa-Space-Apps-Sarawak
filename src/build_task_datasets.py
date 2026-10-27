"""Build per-task modelling tables from the verified NASA FLEX experimental table.

Input : data/processed/dashboard_records.csv (FLEX rows, copied unchanged from
        data/raw/nasa_psi/PSI-69_Experimental table_FLEX.csv)
Output: data/processed/tasks/flex_tests.csv          one row per FLEX test, task-ready columns
        data/processed/tasks/flex_exclusions.csv     every row/task exclusion with its reason
        data/processed/tasks/flex_tests.metadata.json definitions, units, rules, counts, input checksum

Column meanings and units follow NASA/TP-2015-216046, Appendix A.1 and Table IX:
  "Burning rate" = average burning-rate constant from a linear fit of droplet diameter squared vs time, mm^2/s
  the PSI CSV headers "O", "N", "CO", "He" are the ambient O2, N2, CO2 and He mole fractions
  "-" (stored as an en dash) = no value
Only quantities known before a test starts are used as model inputs; burn time, extinction diameter and
the test outcome are results of the same burn.

Run:  python -m src.build_task_datasets
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from app.services.psi_parsing import numeric_cell

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data" / "processed" / "dashboard_records.csv"
OUT_DIR = ROOT / "data" / "processed" / "tasks"

LABEL_MAP = {"Extinction": 1, "Completion": 0}  # Disruption and anything else: excluded from the label
PRESSURE_MIN_MMHG = 1.0                           # 0 mmHg cannot sustain a burn; treated as an invalid entry


def _num(series: pd.Series) -> pd.Series:
    return series.map(lambda v: numeric_cell(v)[0]).astype(float)


def build() -> dict:
    records = pd.read_csv(RECORDS, low_memory=False)
    flex = records[records["dataset_key"] == "FLEX"].copy()
    out = pd.DataFrame({
        "record_id": flex["record_id"],
        "flex_identifier": flex["flex_identifier"].astype(str),
        "group_test_date": flex["test_date"].astype(str),
        "fuel": flex["fuel"].astype(str),
        "is_heptane": (flex["fuel"].astype(str) == "Heptane").astype(int),
        "pressure_mmhg": _num(flex["ambient_pressure_mmhg"]),
        "x_o2": _num(flex["o_initial_ambient_composition_mole_fraction"]),
        "x_n2": _num(flex["n_initial_ambient_composition_mole_fraction"]),
        "x_co2": _num(flex["co_initial_ambient_composition_mole_fraction"]),
        "x_he": _num(flex["he_initial_ambient_composition_mole_fraction"]),
        "d0_mm": _num(flex["droplet_initial_diameter_mm"]),
        "burning_rate_constant_mm2_s": _num(flex["burning_rate_mm"]),
        "test_end": flex["test_end"].astype(str).str.strip(),
    })
    out["extinction_label"] = out["test_end"].map(LABEL_MAP)  # NaN for Disruption / unknown

    exclusions: list[dict] = []

    def exclude(mask: pd.Series, task: str, reason: str) -> None:
        for rid in out.loc[mask, "record_id"]:
            exclusions.append({"record_id": rid, "task": task, "reason": reason})

    dup = out["flex_identifier"].duplicated(keep=False)
    invalid_p = out["pressure_mmhg"].isna() | (out["pressure_mmhg"] < PRESSURE_MIN_MMHG)
    missing_inputs = out[["pressure_mmhg", "x_o2", "x_co2", "x_he", "d0_mm"]].isna().any(axis=1)
    for task in ("flex_droplet_burning_rate_constant", "flex_extinction_outcome"):
        exclude(dup, task, "identifier appears twice with conflicting values; cannot tell which record is correct")
        exclude(invalid_p & ~dup, task, f"ambient pressure missing or below {PRESSURE_MIN_MMHG} mmHg (physically invalid entry)")
        exclude(missing_inputs & ~dup & ~invalid_p, task, "a required input (pressure, O2, CO2, He or initial diameter) is missing")
    base_ok = ~dup & ~invalid_p & ~missing_inputs
    exclude(base_ok & out["burning_rate_constant_mm2_s"].isna(), "flex_droplet_burning_rate_constant",
            "burning-rate constant not reported for this test")
    exclude(base_ok & out["extinction_label"].isna(), "flex_extinction_outcome",
            "test ended by droplet disruption (or another non-defined outcome), so neither extinction nor completion was observed")
    out["use_burning_rate_task"] = base_ok & out["burning_rate_constant_mm2_s"].notna()
    out["use_extinction_task"] = base_ok & out["extinction_label"].notna()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "flex_tests.csv", index=False)
    pd.DataFrame(exclusions, columns=["record_id", "task", "reason"]).to_csv(OUT_DIR / "flex_exclusions.csv", index=False)
    meta = {
        "input": str(RECORDS.relative_to(ROOT).as_posix()),
        "input_sha256": hashlib.sha256(RECORDS.read_bytes()).hexdigest(),
        "source_table": "data/raw/nasa_psi/PSI-69_Experimental table_FLEX.csv (verified, see reports/data_provenance_audit.json)",
        "definitions_source": "NASA/TP-2015-216046 (https://ntrs.nasa.gov/citations/20150023456), Appendix A.1 and Table IX",
        "columns": {
            "burning_rate_constant_mm2_s": "average burning-rate constant from a linear fit of droplet diameter squared vs time (mm^2/s)",
            "x_o2 / x_n2 / x_co2 / x_he": "ambient mole fractions; PSI CSV headers 'O', 'N', 'CO', 'He' are O2, N2, CO2, He",
            "d0_mm": "initial droplet diameter (mm)",
            "pressure_mmhg": "absolute ambient pressure (mmHg)",
            "is_heptane": "1 = heptane, 0 = methanol",
            "group_test_date": "test date; tests on the same day share a chamber fill and setup, so a whole day is kept in one partition",
        },
        "label_rule": {"Extinction": 1, "Completion": 0, "Disruption": "excluded (the burn ended by droplet break-up)"},
        "rows": int(len(out)),
        "rows_for_burning_rate_task": int(out["use_burning_rate_task"].sum()),
        "rows_for_extinction_task": int(out["use_extinction_task"].sum()),
        "extinction_class_counts": out.loc[out["use_extinction_task"], "extinction_label"].value_counts().to_dict(),
        "test_dates": int(out["group_test_date"].nunique()),
        "exclusions": int(len(exclusions)),
    }
    (OUT_DIR / "flex_tests.metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, default=str))
