"""Summarize measured MaCFP PMMA cone-calorimeter runs for after-test prediction.

Input is a sparse clone of https://github.com/MaCFP/matl-db.git at
data/raw/macfp-matl-db, specifically PMMA/Calibration_Data. Each accepted CSV
becomes one experiment row. No simulated or default measurements are created.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = ROOT / "data" / "raw" / "macfp-matl-db"
SOURCE_DIR = SOURCE_REPO / "PMMA" / "Calibration_Data"
OUTPUT_DIR = ROOT / "data" / "processed"
DATASET_PATH = OUTPUT_DIR / "macfp_measured_cone_experiments.csv"
EXCLUSIONS_PATH = OUTPUT_DIR / "macfp_measured_cone_exclusions.csv"
METADATA_PATH = OUTPUT_DIR / "macfp_measured_cone_metadata.json"
SOURCE_URL = "https://github.com/MaCFP/matl-db"
BURN_THRESHOLD_KW_M2 = 10.0
MIN_SAMPLES = 50


def _source_commit() -> str:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Measured source files are missing at {SOURCE_DIR}. Clone the "
            f"public MaCFP repository into {SOURCE_REPO}."
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SOURCE_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "PMMA/Calibration_Data"],
        cwd=SOURCE_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("MaCFP calibration files have local edits; source commit would be misleading")
    return commit


def _header_and_units(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = [cell.strip() for cell in next(reader)]
        units = [cell.strip() for cell in next(reader)]
    if len(header) != len(units):
        raise ValueError("header and unit columns differ")
    return header, units


def _unit(unit: str) -> str:
    return re.sub(r"[\s\[\]]", "", unit).lower()


def _summarize_file(path: Path, commit: str) -> dict[str, object]:
    header, units = _header_and_units(path)
    normalized = [column.lower() for column in header]
    time_idx = normalized.index("time")
    hrr_idx = next(
        idx for idx, column in enumerate(normalized) if column in {"hrr", "heat release"}
    )
    if _unit(units[time_idx]) != "s":
        raise ValueError(f"time unit is {units[time_idx]!r}, expected seconds")
    if _unit(units[hrr_idx]) not in {"kw/m2", "kwm-2"}:
        raise ValueError(f"HRR unit is {units[hrr_idx]!r}, expected kW/m2")

    flux_match = re.search(r"_Cone_(\d+)kW(?:_|\.)", path.name, flags=re.IGNORECASE)
    if not flux_match:
        raise ValueError("incident heat flux is missing from experiment filename")
    heat_flux = int(flux_match.group(1))

    measurements = pd.read_csv(path, skiprows=[1], low_memory=False)
    time = pd.to_numeric(measurements[header[time_idx]], errors="coerce")
    hrr = pd.to_numeric(measurements[header[hrr_idx]], errors="coerce")
    valid = np.isfinite(time) & np.isfinite(hrr)
    time = time[valid].to_numpy(dtype=float)
    hrr = hrr[valid].to_numpy(dtype=float)
    if len(time) < MIN_SAMPLES:
        raise ValueError(f"only {len(time)} numeric time/HRR pairs")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time values are not strictly increasing")
    if hrr[0] >= BURN_THRESHOLD_KW_M2:
        raise ValueError("combustion is already above threshold at first observation")
    if hrr[-1] >= BURN_THRESHOLD_KW_M2:
        raise ValueError("combustion remains above threshold at last observation")

    active = np.flatnonzero(hrr >= BURN_THRESHOLD_KW_M2)
    if len(active) < 5:
        raise ValueError("fewer than five measurements above burn threshold")
    start_idx, end_idx = int(active[0]), int(active[-1])
    burn_time = time[start_idx : end_idx + 1]
    burn_hrr = np.maximum(hrr[start_idx : end_idx + 1], 0.0)
    duration = float(burn_time[-1] - burn_time[0])
    if duration <= 0:
        raise ValueError("burn duration is not positive")
    # A time-weighted mean handles experiments sampled at different intervals.
    mean_hrr = float(
        np.sum((burn_hrr[:-1] + burn_hrr[1:]) * np.diff(burn_time) / 2) / duration
    )
    peak_hrr = float(np.max(hrr))
    if mean_hrr <= 0 or peak_hrr <= 0:
        raise ValueError("measured heat release must be positive")

    source_path = path.relative_to(SOURCE_REPO).as_posix()
    lab = path.parent.name
    return {
        "dataset_source": "MaCFP/matl-db PMMA Calibration_Data",
        "material": "PMMA",
        "laboratory": lab,
        "group_id": f"MaCFP_PMMA_{re.sub(r'[^A-Za-z0-9]+', '_', lab)}",
        "experiment_id": path.stem,
        "experiment_file": path.name,
        "source_path": source_path,
        "source_url": f"https://raw.githubusercontent.com/MaCFP/matl-db/{commit}/{quote(source_path)}",
        "source_commit": commit,
        "radiant_heat_flux_kw_m2": heat_flux,
        "real_burn_duration_s": duration,
        "real_mean_hrr_kw_m2": mean_hrr,
        "target_peak_hrr_kw_m2": peak_hrr,
        "burn_start_s": float(burn_time[0]),
        "burn_end_s": float(burn_time[-1]),
        "numeric_time_hrr_points": len(time),
    }


def build_measured_cone_dataset() -> pd.DataFrame:
    commit = _source_commit()
    paths = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv" and "_cone_" in path.name.lower()
    )
    if not paths:
        raise FileNotFoundError(f"No measured cone-calorimeter CSV files under {SOURCE_DIR}")

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(_summarize_file(path, commit))
        except (ValueError, KeyError, IndexError, StopIteration, pd.errors.ParserError) as exc:
            exclusions.append({"source_path": path.relative_to(SOURCE_REPO).as_posix(), "reason": str(exc)})

    if not records:
        raise ValueError("No valid measured PMMA cone-calorimeter experiments; no dataset was written")
    result = pd.DataFrame.from_records(records).sort_values("source_path").reset_index(drop=True)
    if result["experiment_id"].duplicated().any():
        raise ValueError("Experiment IDs are not unique")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATASET_PATH, index=False, float_format="%.6f", encoding="utf-8")
    pd.DataFrame(exclusions, columns=["source_path", "reason"]).to_csv(
        EXCLUSIONS_PATH, index=False, encoding="utf-8"
    )
    metadata = {
        "source_repo": SOURCE_URL,
        "source_commit": commit,
        "source_scope": "PMMA/Calibration_Data/*/*Cone*.csv measured cone calorimeter runs only",
        "material": "PMMA",
        "test_type": "cone calorimetry",
        "burn_threshold_kw_m2": BURN_THRESHOLD_KW_M2,
        "real_burn_duration_s_definition": "Time from first to last recorded HRR >= 10 kW/m2; runs with HRR >= 10 at the first or final sample are excluded.",
        "real_mean_hrr_kw_m2_definition": "Trapezoidal time-weighted mean of nonnegative measured HRR between burn start and end.",
        "target_peak_hrr_kw_m2_definition": "Maximum recorded numeric HRR over the complete experiment, in kW/m2.",
        "group_id_definition": "Laboratory; keeps all runs from the same institution in one cross-validation fold.",
        "excluded_incomplete_runs": "Runs above the 10 kW/m2 threshold at first or final observation are excluded because full burn duration is not observed.",
        "note": "These are terrestrial PMMA cone tests; they do not directly validate spacecraft or microgravity predictions.",
        "candidate_files": len(paths),
        "included_experiments": len(result),
        "excluded_experiments": len(exclusions),
        "laboratories": int(result["laboratory"].nunique()),
        "peak_hrr_range_kw_m2": [
            float(result["target_peak_hrr_kw_m2"].min()),
            float(result["target_peak_hrr_kw_m2"].max()),
        ],
        "hazard_labels_at_350_kw_m2": {
            "below_350": int((result["target_peak_hrr_kw_m2"] < 350).sum()),
            "at_or_above_350": int((result["target_peak_hrr_kw_m2"] >= 350).sum()),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(result)} measured PMMA experiments from {metadata['laboratories']} labs "
        f"to {DATASET_PATH} ({len(exclusions)} files excluded; see {EXCLUSIONS_PATH})."
    )
    return result


if __name__ == "__main__":
    build_measured_cone_dataset()
