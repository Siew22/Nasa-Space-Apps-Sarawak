from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "config" / "dataset_catalog.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed"
DEMO_DIR = ROOT / "data" / "demo"
# Synthetic demo tables are NOT ingested by default, so they can never mix with the official NASA tables.
INPUT_DIRS = (ROOT / "data" / "raw", ROOT / "data" / "starter_verified")
# Source repositories cloned under data/raw are handled by their own pipelines (e.g. MaCFP).
EXCLUDED_SUBDIRS = (ROOT / "data" / "raw" / "macfp-matl-db",)
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}


def normalize_column_name(name: Any) -> str:
    val = str(name).strip().lower()
    val = val.replace("%", " pct ").replace("#", " number ")
    val = val.replace("₂", "2").replace("₃", "3").replace("₄", "4")
    val = re.sub(r"[^a-z0-9]+", "_", val)
    return val.strip("_") or "unnamed_column"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # NASA's FLEX table is Windows-1252 (0x96 = en dash meaning "no value"); BASS-II is UTF-8 with BOM.
        # Try UTF-8 first and fall back explicitly; values are never altered, only decoded.
        try:
            return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, low_memory=False, encoding="cp1252")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame([payload])
    raise ValueError(f"Unsupported format: {suffix}")


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path, keep_default_na=False)
    catalog["_compiled_regex"] = catalog["filename_regex"].map(re.compile)
    catalog["_pattern_len"] = catalog["filename_regex"].str.len()
    return catalog.sort_values("_pattern_len", ascending=False).reset_index(drop=True)


def match_source(path: Path, catalog: pd.DataFrame) -> dict[str, str]:
    for _, item in catalog.iterrows():
        if item["_compiled_regex"].search(path.name):
            return {k: str(item.get(k, "")) for k in catalog.columns if not k.startswith("_")}
    return {
        "dataset_key": "UNCLASSIFIED",
        "display_name": "Unclassified NASA export",
        "source_page": "",
        "doi": "",
        "license_status": "Not yet verified",
    }


def build(catalog_path: Path = DEFAULT_CATALOG, output_dir: Path = DEFAULT_OUTPUT,
          include_demo: bool = False) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    files = []
    for d in (*INPUT_DIRS, DEMO_DIR) if include_demo else INPUT_DIRS:
        if d.exists():
            files.extend([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                          and not any(ex in p.parents for ex in EXCLUDED_SUBDIRS)])
    files = sorted(set(files))

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    field_rows = []
    report = {"input_file_count": len(files), "files": [], "row_count": 0}

    for path in files:
        relative = path.relative_to(ROOT).as_posix()  # same IDs on Windows and Linux
        try:
            raw = read_table(path)
            if raw.empty:
                continue
            cols = [normalize_column_name(c) for c in raw.columns]
            df = raw.copy()
            df.columns = cols
            meta = match_source(path, catalog)
            if DEMO_DIR in path.parents:
                # Synthetic demo files only borrow a dataset_key so the dashboard can route them to the
                # right track; they must never inherit the real investigation's DOI, page or licence.
                meta = {**meta, "display_name": f"{meta.get('display_name', '')} (SYNTHETIC DEMO)",
                        "source_page": "", "doi": "", "license_status": "Not applicable: synthetic demo data"}

            df["dataset_key"] = meta.get("dataset_key", "UNCLASSIFIED")
            df["dataset_name"] = meta.get("display_name", "Unclassified")
            df["source_file"] = relative
            df["source_url"] = meta.get("source_page", "")
            df["source_doi"] = meta.get("doi", "")
            df["source_license_status"] = meta.get("license_status", "")
            df["source_row_number"] = range(1, len(df) + 1)
            df["record_id"] = df.apply(lambda r: f"{df['dataset_key'].iloc[0]}:{relative}:{r['source_row_number']}", axis=1)

            records.append(df)
            for orig, norm in zip(raw.columns, cols):
                s = df[norm]
                field_rows.append({
                    "dataset_key": meta.get("dataset_key"),
                    "source_file": relative,
                    "original_field_name": str(orig),
                    "field_name": norm,
                    "non_null_values": int(s.notna().sum()),
                    "distinct_non_null_values": int(s.nunique()),
                    "inferred_dtype": str(s.dtype),
                })
            report["files"].append({"file": relative, "status": "loaded", "rows": len(df), "dataset_key": meta.get("dataset_key")})
        except Exception as e:
            report["files"].append({"file": relative, "status": "error", "error": str(e)})

    if records:
        combined = pd.concat(records, ignore_index=True, sort=False)
        meta_cols = ["record_id", "dataset_key", "dataset_name", "source_file", "source_url", "source_doi", "source_license_status", "source_row_number"]
        all_cols = meta_cols + [c for c in combined.columns if c not in meta_cols]
        combined = combined[all_cols]
        combined.to_csv(output_dir / "dashboard_records.csv", index=False, encoding="utf-8-sig")
        report["row_count"] = len(combined)

        summary = combined.groupby(["dataset_key", "dataset_name"]).agg(record_count=("record_id", "count"), source_file_count=("source_file", "nunique")).reset_index()
        summary.to_csv(output_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(field_rows).to_csv(output_dir / "field_catalog.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "ingestion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalise raw experiment tables into data/processed/dashboard_records.csv")
    parser.add_argument("--include-demo", action="store_true",
                        help="also ingest the SYNTHETIC tables in data/demo/ (for demonstrations only)")
    rep = build(include_demo=parser.parse_args().include_demo)
    print(json.dumps(rep, indent=2))