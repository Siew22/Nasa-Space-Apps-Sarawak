"""Evidence-based data provenance audit.

Status of every dataset is COMPUTED here from checks, never typed in:

  SYNTHETIC_DEMO     every row is reproduced exactly by a generator script in this repo
  VERIFIED_SOURCE    rebuilt byte-for-byte from source files at a pinned revision, or the raw
                     file's SHA-256 matches an official checksum registered in config/data_sources.json
                     (for the hand-transcribed Apollo 11 tables: the registered NASA text matches, every
                     transcribed number is found in NASA's OCR of that page, and the tables rebuild identically)
  UNVERIFIED_SOURCE  anything else (e.g. the source files are missing, so nothing can be checked)

Outputs (regenerated on every run):
  reports/data_provenance_audit.json   dataset-level status, evidence codes, what is missing
  reports/provenance_mapping.csv       one row per record: identifiers, source file, fields/units,
                                       transformations, verification status and explanation

Run:  python -m src.provenance_audit
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "config" / "data_sources.json"
RECORDS_PATH = ROOT / "data" / "processed" / "dashboard_records.csv"
FIELD_CATALOG_PATH = ROOT / "data" / "processed" / "field_catalog.csv"
MODEL_METADATA_PATH = ROOT / "models" / "spacecraft_fire_peak_hrr_v1.metadata.json"
LEGACY_DIR = ROOT / "data" / "legacy"
REPORT_PATH = ROOT / "reports" / "data_provenance_audit.json"
MAPPING_PATH = ROOT / "reports" / "provenance_mapping.csv"

META_COLUMNS = {"record_id", "dataset_key", "dataset_name", "source_file", "source_url", "source_doi",
                "source_license_status", "source_row_number", "source_note"}
ORIGINAL_ID_COLUMN = {"BASS_II": "test_number", "FLEX": "flex_identifier"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(code: str, **detail: Any) -> dict[str, Any]:
    return {"code": code, "detail": detail}


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _generator_output(dataset_key: str, sources: dict) -> pd.DataFrame | None:
    """Run the in-repo synthetic generator for this dataset into a temp dir and return its rows."""
    spec = sources["datasets"][dataset_key].get("demo_generator", "")
    if not spec:
        return None
    from src import generate_synthetic_demo_tables as gen
    from src.build_dashboard_data import normalize_column_name

    func = getattr(gen, spec.split(":")[1])
    with tempfile.TemporaryDirectory() as tmp:
        path = func(Path(tmp))
        df = pd.read_csv(path)
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def _rows_reproduced(records: pd.DataFrame, generated: pd.DataFrame) -> int:
    """Count processed rows whose every data value equals the generator's row with the same position."""
    cols = [c for c in records.columns if c not in META_COLUMNS and c in generated.columns
            and records[c].notna().any()]
    if len(records) != len(generated) or not cols:
        return 0
    rec = records.sort_values("source_row_number")[cols].reset_index(drop=True)
    gen = generated[cols].reset_index(drop=True)
    same = np.ones(len(rec), dtype=bool)
    for c in cols:
        a_num, b_num = pd.to_numeric(rec[c], errors="coerce"), pd.to_numeric(gen[c], errors="coerce")
        if a_num.notna().all() and b_num.notna().all():
            same &= np.isclose(a_num.to_numpy(float), b_num.to_numpy(float), rtol=0.0, atol=1e-9)
        else:
            same &= rec[c].astype(str).str.strip().to_numpy() == gen[c].astype(str).str.strip().to_numpy()
    return int(same.sum())


def audit_nasa_format_dataset(key: str, records: pd.DataFrame, sources: dict,
                              catalog: pd.DataFrame) -> tuple[dict, list[dict]]:
    src = sources["datasets"][key]
    rows = records[records["dataset_key"] == key].copy()
    found: list[dict] = []
    id_col = ORIGINAL_ID_COLUMN.get(key)
    source_files = sorted(rows["source_file"].dropna().unique().tolist())

    generated = _generator_output(key, sources)
    reproduced = _rows_reproduced(rows, generated) if generated is not None else 0
    if generated is not None and reproduced == len(rows) and len(rows) > 0:
        found.append(evidence("GENERATOR_REPRODUCES_ALL_ROWS", generator=src["demo_generator"], rows=len(rows)))
    elif reproduced:
        found.append(evidence("GENERATOR_REPRODUCES_SOME_ROWS", generator=src["demo_generator"],
                              rows=reproduced, of=len(rows)))

    registered = {f["sha256"]: f for f in src.get("official_files", []) if f.get("sha256")}
    checksum_match = []
    for rel in source_files:
        path = ROOT / rel
        if not path.exists():
            found.append(evidence("SOURCE_FILE_MISSING", path=rel))
        elif sha256(path) in registered:
            checksum_match.append(rel)
            found.append(evidence("OFFICIAL_CHECKSUM_MATCH", path=rel))
    if not registered:
        found.append(evidence("NO_OFFICIAL_FILE_REGISTERED", investigation=src.get("investigation_id")))

    dup_rows = int(rows.drop(columns=list(META_COLUMNS & set(rows.columns))).duplicated().sum())
    dup_ids = int(rows[id_col].duplicated().sum()) if id_col in rows else None
    if dup_rows:
        found.append(evidence("DUPLICATE_ROWS", count=dup_rows))

    if any(e["code"] == "GENERATOR_REPRODUCES_ALL_ROWS" for e in found):
        status = "SYNTHETIC_DEMO"
    elif checksum_match and len(checksum_match) == len(source_files) and not reproduced:
        status = "VERIFIED_SOURCE"
    else:
        status = "UNVERIFIED_SOURCE"

    fields = catalog[catalog["dataset_key"] == key][["original_field_name", "field_name"]]
    field_text = "; ".join(f"{o}" for o in fields["original_field_name"]
                           if o not in ("source_url", "source_note"))
    mapping = []
    explain, explain_zh = {
        "SYNTHETIC_DEMO": (f"Values are reproduced exactly by {src.get('demo_generator')}; not a NASA measurement.",
                           f"数值可由 {src.get('demo_generator')} 完全复现；并非 NASA 测量值。"),
        "VERIFIED_SOURCE": ("Raw file matches a registered official checksum.", "原始文件与登记的官方校验值一致。"),
        "UNVERIFIED_SOURCE": ("No authoritative source file could be matched to this record.",
                              "无法为该记录匹配到权威来源文件。"),
    }[status]
    for _, r in rows.sort_values("source_row_number").iterrows():
        mapping.append({
            "internal_record_id": r["record_id"],
            "dataset_key": key,
            "investigation": f"{src.get('investigation_id')} {src.get('display_name')}",
            "original_identifier": r.get(id_col, ""),
            "source_file": r.get("source_file", ""),
            "source_file_exists": (ROOT / str(r.get("source_file", ""))).exists(),
            "source_row_number": r.get("source_row_number", ""),
            "persistent_identifier": "" if status != "VERIFIED_SOURCE" else src.get("real_source_doi", ""),
            "real_investigation_doi_for_reference": src.get("real_source_doi", ""),
            "fields_and_units_as_in_file": field_text,
            "transformations": "column names normalised to snake_case; values copied unchanged (src/build_dashboard_data.py)",
            "verification_status": status,
            "explanation": explain,
            "explanation_zh": explain_zh,
        })
    inv, page = src.get("investigation_id"), src.get("real_source_page")
    summary = {
        "status": status,
        "result_type": "SYNTHETIC_DEMO" if status == "SYNTHETIC_DEMO" else ("REAL_MEASUREMENT" if status == "VERIFIED_SOURCE" else "UNVERIFIED"),
        "rows": int(len(rows)),
        "unique_original_identifiers": int(rows[id_col].nunique()) if id_col in rows else None,
        "duplicate_identifiers": dup_ids,
        "independent_experiments_verified": int(len(rows)) if status == "VERIFIED_SOURCE" else 0,
        "source_files": source_files,
        "evidence": found,
        "required_to_verify": [] if status == "VERIFIED_SOURCE" else [
            f"Download the experimental table for {inv} from {page}",
            "Store it unchanged under data/raw/ and register its SHA-256, download URL and date in config/data_sources.json",
            "Map each column's definition and unit from the PSI documentation into config/target_registry.json",
            "Rebuild data/processed/dashboard_records.csv and rerun src/provenance_audit.py",
        ],
        "required_to_verify_zh": [] if status == "VERIFIED_SOURCE" else [
            f"从 {page} 下载 {inv} 的实验数据表",
            "原样保存到 data/raw/，并在 config/data_sources.json 中登记其 SHA-256、下载网址与日期",
            "依据 PSI 文档在 config/target_registry.json 中填写各列的定义与单位",
            "重新构建 data/processed/dashboard_records.csv 并重新运行 src/provenance_audit.py",
        ],
    }
    return summary, mapping


def audit_macfp(sources: dict) -> tuple[dict, list[dict]]:
    src = sources["datasets"]["MACFP_PMMA_CONE"]
    clone = ROOT / src["local_clone"]
    processed = ROOT / src["processed_file"]
    found: list[dict] = []
    status = "UNVERIFIED_SOURCE"
    rebuilt_hash = None
    commit_ok = clean_ok = False
    if not processed.exists():
        found.append(evidence("PROCESSED_FILE_MISSING", path=src["processed_file"]))
        return {"status": status, "result_type": "UNVERIFIED", "rows": 0, "evidence": found,
                "required_to_verify": ["Run src/build_measured_cone_dataset.py"]}, []
    committed_hash = sha256(processed)
    df = pd.read_csv(processed)

    if (clone / src["source_subdir"]).is_dir():
        commit = _git(["rev-parse", "HEAD"], clone)
        commit_ok = commit == src["pinned_commit"]
        clean_ok = _git(["status", "--porcelain", "--", src["source_subdir"]], clone) == ""
        found.append(evidence("CLONE_AT_PINNED_COMMIT" if commit_ok else "CLONE_COMMIT_MISMATCH",
                              commit=commit, pinned=src["pinned_commit"]))
        found.append(evidence("CLONE_CLEAN" if clean_ok else "CLONE_HAS_LOCAL_EDITS"))
        import src.build_measured_cone_dataset as builder

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            saved = (builder.OUTPUT_DIR, builder.DATASET_PATH, builder.EXCLUSIONS_PATH, builder.METADATA_PATH)
            builder.OUTPUT_DIR = tmp
            builder.DATASET_PATH = tmp / "rebuilt.csv"
            builder.EXCLUSIONS_PATH = tmp / "excl.csv"
            builder.METADATA_PATH = tmp / "meta.json"
            try:
                builder.build_measured_cone_dataset()
                rebuilt_hash = sha256(builder.DATASET_PATH)
            finally:
                builder.OUTPUT_DIR, builder.DATASET_PATH, builder.EXCLUSIONS_PATH, builder.METADATA_PATH = saved
        found.append(evidence("REBUILD_BYTE_IDENTICAL" if rebuilt_hash == committed_hash else "REBUILD_DIFFERS",
                              rebuilt_sha256=rebuilt_hash, committed_sha256=committed_hash))
    else:
        found.append(evidence("SOURCE_CLONE_MISSING", path=src["local_clone"]))

    if MODEL_METADATA_PATH.exists():
        trained_on = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))["training_data"]["csv_sha256"]
        found.append(evidence("MODEL_TRAINED_ON_THIS_FILE" if trained_on == committed_hash else "MODEL_TRAINED_ON_OTHER_FILE",
                              model_csv_sha256=trained_on))
    dup = int(df.duplicated(["radiant_heat_flux_kw_m2", "real_burn_duration_s", "real_mean_hrr_kw_m2",
                             "target_peak_hrr_kw_m2"]).sum())
    if dup:
        found.append(evidence("DUPLICATE_ROWS", count=dup))
    if commit_ok and clean_ok and rebuilt_hash == committed_hash:
        status = "VERIFIED_SOURCE"

    mapping = []
    for _, r in df.iterrows():
        exists = (clone / r["source_path"]).exists()
        mapping.append({
            "internal_record_id": r["experiment_id"],
            "dataset_key": "MACFP_PMMA_CONE",
            "investigation": "MaCFP/matl-db PMMA Calibration_Data (ground-based cone calorimeter)",
            "original_identifier": r["experiment_file"],
            "source_file": f"{src['local_clone']}/{r['source_path']}",
            "source_file_exists": exists,
            "source_row_number": "",
            "persistent_identifier": r["source_url"],
            "real_investigation_doi_for_reference": "",
            "fields_and_units_as_in_file": "Time [s]; HRR [kW/m2] (unit row checked by the builder)",
            "transformations": ("burn window = HRR >= 10 kW/m2; real_burn_duration_s = window length; "
                                "real_mean_hrr_kw_m2 = trapezoidal mean in window; target_peak_hrr_kw_m2 = max HRR; "
                                "heat flux parsed from file name"),
            "verification_status": status if exists or status != "VERIFIED_SOURCE" else "UNVERIFIED_SOURCE",
            "explanation": ("Rebuilt byte-for-byte from the source file at the pinned MaCFP commit."
                            if status == "VERIFIED_SOURCE" else "Source clone unavailable or rebuild differs."),
            "explanation_zh": ("已由固定 MaCFP 提交中的来源文件逐字节重建。"
                               if status == "VERIFIED_SOURCE" else "来源克隆不可用或重建结果不一致。"),
        })
    summary = {
        "status": status,
        "result_type": "REAL_MEASUREMENT" if status == "VERIFIED_SOURCE" else "UNVERIFIED",
        "rows": int(len(df)),
        "unique_original_identifiers": int(df["experiment_id"].nunique()),
        "independent_groups": int(df["laboratory"].nunique()),
        "group_definition": "laboratory",
        "duplicate_identifiers": int(df["experiment_id"].duplicated().sum()),
        "independent_experiments_verified": int(len(df)) if status == "VERIFIED_SOURCE" else 0,
        "source_commit": src["pinned_commit"],
        "evidence": found,
        "required_to_verify": [] if status == "VERIFIED_SOURCE" else [
            f"Clone {src['source_repo']} into {src['local_clone']} at commit {src['pinned_commit']} and rerun this audit"],
        "required_to_verify_zh": [] if status == "VERIFIED_SOURCE" else [
            f"将 {src['source_repo']} 克隆到 {src['local_clone']}，切换到提交 {src['pinned_commit']} 后重新运行本审计"],
    }
    return summary, mapping


def audit_apollo_ascent(sources: dict) -> tuple[dict, list[dict]]:
    """Apollo 11 launch values from NASA/SP-2000-4029: official NTRS text checksum, transcription checked against
    NASA's own OCR of the page, consistency across tables, agreement with the primary report, byte-identical rebuild."""
    import src.build_rocket_ascent as builder

    key = "APOLLO11_ASCENT"
    src = sources["datasets"][key]
    entry = src["official_files"][0]
    found: list[dict] = []
    official = ROOT / entry["path"]
    checksum_ok = official.exists() and official.stat().st_size == entry["size_bytes"] and sha256(official) == entry["sha256"]
    found.append(evidence("OFFICIAL_CHECKSUM_MATCH" if checksum_ok else
                          ("SOURCE_FILE_MISSING" if not official.exists() else "CHECKSUM_MISMATCH"), path=entry["path"]))
    for d in src.get("derived_files", []):
        p = ROOT / d["path"]
        found.append(evidence("PAGE_IMAGE_MATCH" if p.exists() and sha256(p) == d["sha256"] else "PAGE_IMAGE_MISSING_OR_CHANGED",
                              path=d["path"]))

    checks = builder.run_checks()
    ocr = checks.get("ocr", {})
    ocr_ok = bool(ocr.get("passed"))
    found.append(evidence("TRANSCRIPTION_MATCHES_OCR" if ocr_ok else "TRANSCRIPTION_NOT_CONFIRMED",
                          values=ocr.get("values_checked"), times=ocr.get("times_checked"),
                          missing=(ocr.get("values_missing_from_ocr", []) + ocr.get("times_missing_from_ocr", []))))
    internal_ok = checks["internal"]["passed"]
    found.append(evidence("TIMES_CONSISTENT_ACROSS_TABLES" if internal_ok else "TIMES_INCONSISTENT",
                          shared_times=checks["internal"]["shared_times_checked"],
                          ignition_checks=len(checks["internal"]["ignition_vs_burn_duration"])))
    primary = checks["primary_report"]
    found.append(evidence("PRIMARY_REPORT_AGREES", agree=primary["agree"], compared=primary["compared"]))
    for c in primary["conflicts"]:
        found.append(evidence("SECONDARY_SOURCE_CONFLICT", item=c["primary"], sp4029=c["ours"], primary_report=c["primary_value"]))

    outputs = (builder.ASCENT_OUT, builder.TIMELINE_OUT, builder.ORBIT_OUT, builder.PRIMARY_OUT)
    committed = {p.name: sha256(p) if p.exists() else None for p in outputs}
    rebuilt: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        saved = (builder.OUTPUT_DIR, *outputs, builder.METADATA_OUT)
        builder.OUTPUT_DIR = tmp
        builder.ASCENT_OUT, builder.TIMELINE_OUT, builder.ORBIT_OUT, builder.PRIMARY_OUT = (tmp / p.name for p in outputs)
        builder.METADATA_OUT = tmp / "meta.json"
        try:
            builder.build()
            rebuilt = {p.name: sha256(tmp / p.name) for p in outputs}
        except Exception as exc:  # noqa: BLE001 - a failed rebuild is evidence, not a crash
            found.append(evidence("REBUILD_FAILED", error=str(exc)))
        finally:
            (builder.OUTPUT_DIR, builder.ASCENT_OUT, builder.TIMELINE_OUT, builder.ORBIT_OUT, builder.PRIMARY_OUT,
             builder.METADATA_OUT) = saved
    rebuild_ok = bool(rebuilt) and rebuilt == committed
    if rebuilt:
        found.append(evidence("REBUILD_FILES_IDENTICAL" if rebuild_ok else "REBUILD_DIFFERS", files=len(rebuilt),
                              differing=[n for n in rebuilt if rebuilt[n] != committed[n]]))

    if rebuild_ok:
        from app.services import rocket_ascent

        rocket_ascent.load_tables.cache_clear()
        physics = rocket_ascent.validation_checks()
        passed = sum(c["passed"] for c in physics)
        found.append(evidence("PHYSICS_REPRODUCES_NASA" if passed == len(physics) else "PHYSICS_CHECK_FAILED",
                              passed=passed, of=len(physics)))
    for note in json.loads(builder.METADATA_OUT.read_text(encoding="utf-8"))["notes"] if builder.METADATA_OUT.exists() else []:
        if note["code"] in ("HEADER_UNIT_MISPRINT", "SUSPECT_VALUE"):
            found.append(evidence(note["code"], detail=note["detail"]))

    status = "VERIFIED_SOURCE" if checksum_ok and ocr_ok and internal_ok and rebuild_ok else "UNVERIFIED_SOURCE"
    explain, explain_zh = (
        ("Transcribed from NASA/SP-2000-4029; checked against NASA's OCR of the page; processed tables rebuild byte-for-byte.",
         "转录自 NASA/SP-2000-4029；已与 NASA 对该页的 OCR 文本核对，并可逐字节重建。")
        if status == "VERIFIED_SOURCE" else
        ("The transcription could not be confirmed against the registered NASA file.", "无法用登记的 NASA 文件确认该转录。"))
    tables = {"ascent": (builder.ASCENT_OUT, 103, "Apollo 11 Ascent Phase"),
              "orbit": (builder.ORBIT_OUT, 103, "Apollo 11 Earth Orbit Phase"),
              "timeline": (builder.TIMELINE_OUT, 105, "Apollo 11 Timeline")}
    mapping = []
    for table, (path, page, title) in tables.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        for i, r in enumerate(frame.itertuples(index=False), start=1):
            mapping.append({
                "internal_record_id": f"{key}:{table}:{r.event_id}",
                "dataset_key": key,
                "investigation": f"{src['publication']}, printed page {page}, '{title}'",
                "original_identifier": r.event,
                "source_file": f"data/transcribed/{Path(path).name}",
                "source_file_exists": True,
                "source_row_number": i,
                "persistent_identifier": src["real_source_page"],
                "real_investigation_doi_for_reference": "",
                "fields_and_units_as_in_file": "GET hhh:mm:ss; altitude, range n mi; velocities ft/s; angles deg",
                "transformations": "GET -> seconds; n mi x 1.852 -> km; ft/s x 0.3048 -> m/s (src/build_rocket_ascent.py)",
                "verification_status": status,
                "explanation": explain,
                "explanation_zh": explain_zh,
            })
    summary = {
        "status": status,
        "result_type": "REAL_MEASUREMENT" if status == "VERIFIED_SOURCE" else "UNVERIFIED",
        "rows": len(mapping),
        "result_note": "Post-flight values reported by NASA. Rows marked 'theoretical' and the spent-stage apex rows are "
                       "NASA calculations, not tracking measurements.",
        "source_files": [entry["path"]],
        "evidence": found,
        "required_to_verify": [] if status == "VERIFIED_SOURCE" else [
            "Run python -m src.fetch_nasa_psi_tables to restore the registered NTRS text file",
            "Rerun python -m src.build_rocket_ascent and this audit"],
        "required_to_verify_zh": [] if status == "VERIFIED_SOURCE" else [
            "运行 python -m src.fetch_nasa_psi_tables 恢复登记的 NTRS 文本文件",
            "重新运行 python -m src.build_rocket_ascent 与本审计"],
    }
    return summary, mapping


def audit_legacy_files(sources: dict) -> dict[str, dict]:
    """Classify the retired files in data/legacy/ with concrete checks."""
    out: dict[str, dict] = {}
    curated = LEGACY_DIR / "research_curated_dataset.csv"
    synthetic_bass = _generator_output("BASS_II", sources)  # the retired dataset was built from these rows
    if curated.exists() and synthetic_bass is not None:
        c = pd.read_csv(curated, encoding="utf-8-sig")
        bass = c[c["dataset_source"] == "BASS_II"].copy()
        fan = synthetic_bass.set_index("sample_number")["fan_display"]
        sample_no = bass["experiment_group"].str.replace("BASS_SAMPLE_", "", regex=False).astype(float)
        expected = (fan.reindex(sample_no).to_numpy() * 0.0016).clip(min=0.005) * 1.5
        computed = ((bass["target_rate_mm_s"].to_numpy() == 0) |
                    (abs(bass["target_rate_mm_s"].to_numpy() - expected.round(3)) < 1e-9))
        out["data/legacy/research_curated_dataset.csv"] = {
            "status": "SYNTHETIC_DEMO",
            "evidence": [
                evidence("DERIVED_FROM_SYNTHETIC_TABLES", rows=int(len(c))),
                evidence("TARGET_COMPUTED_FROM_INPUT", rows=int(computed.sum()), of=int(len(bass)),
                         formula="fan_display x 0.0016 (min 0.005) x 1.5, or 0 when labelled extinguished"),
                evidence("INCOMPATIBLE_TARGETS_MERGED",
                         detail="BASS-II proxy spread value and FLEX burning rate share column target_rate_mm_s"),
            ],
        }
    for name, cols in (("macfp_empirical_fire_properties.csv", ["radiant_heat_flux_kw_m2", "real_peak_hrr_kw_m2"]),
                       ("spacecraft_fire_benchmark.csv", ["radiant_heat_flux_kw_m2", "target_peak_hrr_kw_m2"])):
        path = LEGACY_DIR / name
        if path.exists():
            d = pd.read_csv(path)
            distinct = {c: int(d[c].nunique()) for c in cols if c in d}
            out[f"data/legacy/{name}"] = {
                "status": "SYNTHETIC_DEMO",
                "evidence": [evidence("CONSTANT_FALLBACK_VALUES", rows=int(len(d)), distinct_values=distinct)],
            }
    meta, template = LEGACY_DIR / "research_curated_dataset.metadata.json", LEGACY_DIR / "research_curated_dataset.metadata.template.json"
    if meta.exists() and template.exists():
        text = meta.read_text(encoding="utf-8")
        out["data/legacy/research_curated_dataset.metadata.json"] = {
            "status": "UNFILLED_TEMPLATE",
            "evidence": [evidence("IDENTICAL_TO_TEMPLATE", identical=meta.read_bytes() == template.read_bytes(),
                                  placeholders=text.count("REPLACE_WITH_"))],
        }
    log = LEGACY_DIR / "logs of fire model.txt"
    if log.exists():
        missing = not (ROOT / "models" / "spacecraft_fire_multitask_v1.keras").exists()
        out["data/legacy/logs of fire model.txt"] = {
            "status": "UNVERIFIABLE",
            "evidence": [evidence("REFERENCED_MODEL_MISSING", model="models/spacecraft_fire_multitask_v1.keras",
                                  missing=missing)],
        }
    return out


def _git_commit() -> str:
    try:
        return _git(["rev-parse", "HEAD"], ROOT)
    except Exception:  # noqa: BLE001 - provenance of the audit itself is best-effort
        return "unknown"


def run() -> dict:
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    records = pd.read_csv(RECORDS_PATH, low_memory=False) if RECORDS_PATH.exists() else pd.DataFrame()
    catalog = pd.read_csv(FIELD_CATALOG_PATH) if FIELD_CATALOG_PATH.exists() else pd.DataFrame(
        columns=["dataset_key", "original_field_name", "field_name"])
    datasets, mapping = {}, []
    for key in ("BASS_II", "FLEX"):
        if not records.empty and (records["dataset_key"] == key).any():
            summary, rows = audit_nasa_format_dataset(key, records, sources, catalog)
        else:
            summary, rows = {"status": "UNVERIFIED_SOURCE", "result_type": "UNVERIFIED", "rows": 0,
                             "evidence": [evidence("NO_RECORDS_LOADED")]}, []
        datasets[key] = summary
        mapping += rows
    datasets["MACFP_PMMA_CONE"], macfp_rows = audit_macfp(sources)
    mapping += macfp_rows
    datasets["APOLLO11_ASCENT"], apollo_rows = audit_apollo_ascent(sources)
    mapping += apollo_rows

    report = {
        "report": "data_provenance_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "src/provenance_audit.py",
        "repository_commit": _git_commit(),
        "status_rules": {
            "SYNTHETIC_DEMO": "Every row is reproduced exactly by a generator script in this repository.",
            "VERIFIED_SOURCE": "Rebuilt byte-for-byte from source files at a pinned revision, or raw file SHA-256 matches a registered official checksum.",
            "UNVERIFIED_SOURCE": "Neither condition holds; the data cannot be traced to an authoritative source.",
        },
        "datasets": datasets,
        "legacy_files": audit_legacy_files(sources),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(mapping).to_csv(MAPPING_PATH, index=False, encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run()
    for key, d in rep["datasets"].items():
        print(f"{key:16s} {d['status']:18s} rows={d.get('rows')}  evidence={[e['code'] for e in d['evidence']]}")
    for path, d in rep["legacy_files"].items():
        print(f"{path}: {d['status']}  {[e['code'] for e in d['evidence']]}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and {MAPPING_PATH.relative_to(ROOT)}")
