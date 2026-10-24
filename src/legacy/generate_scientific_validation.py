"""Audit an empirically curated combustion table and optionally run grouped CV.

The script intentionally refuses to report model performance when the curated
CSV, label definitions, source lineage, or leakage review is absent. Empty CSV
outputs in a blocked run are headers only; they do not imply evaluation ran.

Default input:
  data/processed/research_curated_dataset.csv
Companion metadata:
  data/processed/research_curated_dataset.metadata.json

Use --allow-unverified-provenance only for explicitly exploratory, non-claimable
runs. Do not use exploratory metrics in a scientific claim or competition pitch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "scientific_validation.json"
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "research_curated_dataset.csv"
REPORTS_DIR = ROOT / "reports"

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    VALIDATION_CONFIG = json.load(handle)

FEATURE_COLS: list[str] = VALIDATION_CONFIG["feature_columns"]
TARGET_REG = VALIDATION_CONFIG["regression_target"]
TARGET_CLS = VALIDATION_CONFIG["classification_target"]
REQUIRED_COLUMNS = ["dataset_source", "experiment_group", *FEATURE_COLS, TARGET_REG, TARGET_CLS]

MANIFEST_COLUMNS = [
    "eval_id", "seed", "fold", "record_id", "record_index", "group_id",
    "dataset_source", "partition", "n_train_records", "n_holdout_records",
    "n_train_groups", "n_holdout_groups", "group_overlap_count",
]
METRIC_COLUMNS = [
    "eval_id", "seed", "fold", "model_family", "n_train_records", "n_holdout_records",
    "n_train_groups", "n_holdout_groups", "holdout_class_0", "holdout_class_1",
    "classification_holdout_has_both_classes", "mae_mm_s", "rmse_mm_s", "r2_score",
    "balanced_accuracy", "macro_f1", "evaluation_note",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _empty_result_csvs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(
        REPORTS_DIR / "experiment_split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=METRIC_COLUMNS).to_csv(
        REPORTS_DIR / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )


def _feature_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            result[column] = {"present": False}
            continue
        raw = pd.to_numeric(df[column], errors="coerce")
        finite = raw[np.isfinite(raw.to_numpy(dtype=float, na_value=np.nan))]
        finite_check_values = raw.dropna().to_numpy(dtype=float)
        result[column] = {
            "present": True,
            "missing_or_non_numeric": int(raw.isna().sum()),
            "non_finite": int((~np.isfinite(finite_check_values)).sum()),
            "n_unique_non_missing": int(raw.nunique(dropna=True)),
            "mean": float(finite.mean()) if len(finite) else None,
            "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "variance": float(finite.var(ddof=1)) if len(finite) > 1 else None,
            "min": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return result


def _base_report(data_path: Path, metadata_path: Path) -> dict[str, Any]:
    return {
        "report_schema_version": "1.0",
        "dataset_name": "User-curated combustion research dataset; name/source must be substantiated by companion metadata",
        "created_by_script": "src/generate_scientific_validation.py",
        "data_file": str(data_path),
        "metadata_file": str(metadata_path),
        "audit_status": "not_run",
        "model_evaluation_status": "not_run",
        "scientific_interpretation": "No predictive claim is supported unless source, target, grouping, and leakage checks pass.",
        "software_versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "blockers": [],
        "warnings": [],
        "provenance_and_target_leakage": {
            "automatic_status": "not_provable_from_csv_alone",
            "note": "Software checks structural conditions. The meaning and origin of targets/features and the absence of rule-derived label leakage require an explicit, reviewable metadata declaration and scientific inspection.",
        },
    }


def _metadata_blockers(metadata: dict[str, Any] | None, sources: list[str]) -> list[str]:
    blockers: list[str] = []
    if metadata is None:
        return [
            "Companion metadata file is missing. Copy data/processed/research_curated_dataset.metadata.template.json to research_curated_dataset.metadata.json and fill it with verified information."
        ]
    if metadata.get("data_status") != "curated_empirical_measurements":
        blockers.append("metadata.data_status must explicitly be 'curated_empirical_measurements'.")
    if metadata.get("contains_synthetic_rows") is not False:
        blockers.append("metadata.contains_synthetic_rows must explicitly be false; do not mix synthetic/replicated rows into empirical evaluation.")
    if not str(metadata.get("record_level_definition", "")).strip() or "REPLACE_" in str(metadata.get("record_level_definition", "")):
        blockers.append("metadata.record_level_definition must explain what one row represents.")
    if not str(metadata.get("group_definition", "")).strip() or "REPLACE_" in str(metadata.get("group_definition", "")):
        blockers.append("metadata.group_definition must explain which samples come from the same physical experiment/run.")

    source_metadata = metadata.get("source_datasets")
    if not isinstance(source_metadata, dict):
        blockers.append("metadata.source_datasets must map each dataset_source to its exact source URL, license state, and source file.")
        source_metadata = {}
    for source in sources:
        item = source_metadata.get(source)
        if not isinstance(item, dict):
            blockers.append(f"metadata.source_datasets lacks source '{source}'.")
            continue
        for field in ("source_url", "license_status", "source_file"):
            value = str(item.get(field, "")).strip()
            if not value or "REPLACE_" in value:
                blockers.append(f"metadata.source_datasets[{source!r}].{field} is missing or still a template placeholder.")

    source_target_map = metadata.get("source_target_mapping")
    if not isinstance(source_target_map, dict):
        blockers.append("metadata.source_target_mapping must state how each target is obtained for each source; do not infer target compatibility from matching column names.")
        source_target_map = {}
    for source in sources:
        mapping = source_target_map.get(source)
        if not isinstance(mapping, dict):
            blockers.append(f"metadata.source_target_mapping lacks source '{source}'.")
            continue
        for target in (TARGET_REG, TARGET_CLS):
            mapping_item = mapping.get(target)
            if not isinstance(mapping_item, dict):
                blockers.append(f"metadata.source_target_mapping[{source!r}] lacks target '{target}'.")
                continue
            for field in ("status", "field_or_method", "unit_conversion"):
                value = str(mapping_item.get(field, "")).strip()
                if not value or "REPLACE_" in value:
                    blockers.append(f"metadata.source_target_mapping[{source!r}][{target!r}].{field} is missing or still a template placeholder.")
            if str(mapping_item.get("status", "")).strip().lower() in {"not_available", "unavailable", "none"}:
                blockers.append(f"Target '{target}' is declared unavailable for source '{source}'; do not fill it synthetically to form a joint training table.")

    feature_meta = metadata.get("features")
    if not isinstance(feature_meta, dict):
        blockers.append("metadata.features must define every modeled feature, its unit, and its derivation.")
        feature_meta = {}
    for feature in FEATURE_COLS:
        item = feature_meta.get(feature)
        if not isinstance(item, dict):
            blockers.append(f"metadata.features lacks '{feature}'.")
            continue
        for field in ("definition", "unit", "derivation"):
            value = str(item.get(field, "")).strip()
            if not value or "REPLACE_" in value:
                blockers.append(f"metadata.features[{feature!r}].{field} is missing or still a template placeholder.")

    target_meta = metadata.get("target_definitions")
    if not isinstance(target_meta, dict):
        blockers.append("metadata.target_definitions must document both target definitions.")
        target_meta = {}
    for target in (TARGET_REG, TARGET_CLS):
        item = target_meta.get(target)
        if not isinstance(item, dict):
            blockers.append(f"metadata.target_definitions lacks '{target}'.")
            continue
        required_fields = ("definition", "unit", "measurement_method")
        if target == TARGET_CLS:
            required_fields = (*required_fields, "positive_class")
        for field in required_fields:
            value = str(item.get(field, "")).strip()
            if not value or "REPLACE_" in value:
                blockers.append(f"metadata.target_definitions[{target!r}].{field} is missing or still a template placeholder.")
        if target == TARGET_REG:
            method = str(item.get("empirical_or_derived", "")).strip()
            if method not in {"empirical", "derived"}:
                blockers.append("metadata.target_definitions['target_rate_mm_s'].empirical_or_derived must be 'empirical' or 'derived', with method explained.")

    leakage = metadata.get("leakage_review")
    if not isinstance(leakage, dict):
        blockers.append("metadata.leakage_review must document a human review of synthetic labels and target-derived features.")
    else:
        if leakage.get("reviewer_completed") is not True:
            blockers.append("A human leakage review has not been marked complete in metadata.")
        if leakage.get("no_target_derived_features") is not True:
            blockers.append("Metadata must attest, after review, that no feature improperly encodes the prediction target.")
        if leakage.get("no_synthetic_targets") is not True:
            blockers.append("Metadata must attest, after review, that targets are not synthetic rule labels.")
        for field in ("reviewer", "review_notes"):
            value = str(leakage.get(field, "")).strip()
            if not value or "REPLACE_" in value:
                blockers.append(f"metadata.leakage_review.{field} must identify the reviewer and document the review.")
    return blockers


def _make_regressor(kind: str, seed: int) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    if kind == "Dummy_Baseline":
        estimator = DummyRegressor(strategy="mean")
    else:
        rf_cfg = VALIDATION_CONFIG["random_forest"]
        estimator = RandomForestRegressor(
            n_estimators=int(rf_cfg["n_estimators"]),
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            n_jobs=int(rf_cfg["n_jobs"]),
            random_state=seed,
        )
    return Pipeline([("imputer", imputer), ("estimator", estimator)])


def _make_classifier(kind: str, seed: int) -> Pipeline:
    imputer = SimpleImputer(strategy="median")
    if kind == "Dummy_Baseline":
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        rf_cfg = VALIDATION_CONFIG["random_forest"]
        estimator = RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
            n_jobs=int(rf_cfg["n_jobs"]),
            random_state=seed,
            class_weight="balanced",
        )
    return Pipeline([("imputer", imputer), ("estimator", estimator)])


def _build_tf_multitask_model(input_dim: int, seed: int):
    # Import lazily; standard audits do not need TensorFlow or a GPU.
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # A prior TF operation may have initialized this device; training can proceed.
            pass

    nn_cfg = VALIDATION_CONFIG["neural_network"]
    reg = float(nn_cfg["l2_regularization"])
    inputs = keras.Input(shape=(input_dim,), name="empirical_features")
    x = inputs
    for index, units in enumerate(nn_cfg["hidden_units"]):
        x = layers.Dense(
            int(units), activation="relu",
            kernel_regularizer=keras.regularizers.l2(reg),
            name=f"shared_dense_{index + 1}",
        )(x)
    rate_output = layers.Dense(1, activation="linear", name="spread_pred")(x)
    extinction_output = layers.Dense(1, activation="sigmoid", name="ext_prob")(x)
    model = keras.Model(inputs=inputs, outputs=[rate_output, extinction_output])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(nn_cfg["learning_rate"])),
        loss={"spread_pred": "mse", "ext_prob": "binary_crossentropy"},
    )
    return model


def _run_cv(df: pd.DataFrame, include_tensorflow: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = df["experiment_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_count = len(unique_groups)
    n_splits = int(VALIDATION_CONFIG["n_splits"])
    seeds = [int(value) for value in VALIDATION_CONFIG["seed_list"]]
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y_reg = pd.to_numeric(df[TARGET_REG], errors="raise").to_numpy(dtype=np.float64)
    y_cls = pd.to_numeric(df[TARGET_CLS], errors="raise").to_numpy(dtype=np.int32)
    record_ids = df["record_id"].astype(str).to_numpy() if "record_id" in df else np.arange(len(df)).astype(str)
    sources = df["dataset_source"].astype(str).to_numpy()

    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    eval_id = 0
    use_tf = include_tensorflow and group_count >= int(VALIDATION_CONFIG["neural_network"]["minimum_independent_groups"])
    tf_skip_reason = None
    if include_tensorflow and not use_tf:
        tf_skip_reason = (
            f"Skipped TensorFlow neural network because there are {group_count} independent groups; "
            f"configured minimum is {VALIDATION_CONFIG['neural_network']['minimum_independent_groups']}."
        )
    if use_tf:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            use_tf = False
            tf_skip_reason = "TensorFlow is not installed in this environment; baseline and Random Forest evaluation will still run."

    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled_unique_groups = rng.permutation(unique_groups)
        group_to_idx = {g: i for i, g in enumerate(shuffled_unique_groups)}
        permuted_groups = np.array([group_to_idx[g] for g in groups])

        splitter = GroupKFold(n_splits=n_splits)
        for fold_no, (train_idx, holdout_idx) in enumerate(splitter.split(X, y_cls, groups=permuted_groups), start=1):
            eval_id += 1
            train_groups = set(groups[train_idx])
            holdout_groups = set(groups[holdout_idx])
            group_overlap_count = len(train_groups & holdout_groups)
            if group_overlap_count != 0:
                raise AssertionError(f"Group leakage detected at eval_id={eval_id}: {train_groups & holdout_groups}")

            n_train = len(train_idx)
            n_holdout = len(holdout_idx)
            train_group_count = len(train_groups)
            holdout_group_count = len(holdout_groups)
            for idx in train_idx:
                manifest_rows.append({
                    "eval_id": eval_id, "seed": seed, "fold": fold_no,
                    "record_id": record_ids[idx], "record_index": int(idx),
                    "group_id": groups[idx], "dataset_source": sources[idx],
                    "partition": "cv_train", "n_train_records": n_train,
                    "n_holdout_records": n_holdout, "n_train_groups": train_group_count,
                    "n_holdout_groups": holdout_group_count, "group_overlap_count": group_overlap_count,
                })
            for idx in holdout_idx:
                manifest_rows.append({
                    "eval_id": eval_id, "seed": seed, "fold": fold_no,
                    "record_id": record_ids[idx], "record_index": int(idx),
                    "group_id": groups[idx], "dataset_source": sources[idx],
                    "partition": "cv_holdout", "n_train_records": n_train,
                    "n_holdout_records": n_holdout, "n_train_groups": train_group_count,
                    "n_holdout_groups": holdout_group_count, "group_overlap_count": group_overlap_count,
                })

            y_train_reg, y_holdout_reg = y_reg[train_idx], y_reg[holdout_idx]
            y_train_cls, y_holdout_cls = y_cls[train_idx], y_cls[holdout_idx]
            class_counts = Counter(int(x) for x in y_holdout_cls)
            has_both_classes = len(class_counts) == 2

            for model_name in ("Dummy_Baseline", "Random_Forest"):
                reg_model = _make_regressor(model_name, seed * 100 + fold_no)
                cls_model = _make_classifier(model_name, seed * 100 + fold_no)
                reg_model.fit(X[train_idx], y_train_reg)
                cls_model.fit(X[train_idx], y_train_cls)
                pred_reg = reg_model.predict(X[holdout_idx])
                pred_cls = cls_model.predict(X[holdout_idx]).astype(int)
                metric_rows.append(_metric_row(
                    eval_id, seed, fold_no, model_name, n_train, n_holdout,
                    train_group_count, holdout_group_count, class_counts, has_both_classes,
                    y_holdout_reg, pred_reg, y_holdout_cls, pred_cls,
                ))

            if use_tf:
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                nn_seed = seed * 100 + fold_no
                preprocessor = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
                X_train_nn = preprocessor.fit_transform(X[train_idx])
                X_holdout_nn = preprocessor.transform(X[holdout_idx])
                model = _build_tf_multitask_model(X_train_nn.shape[1], nn_seed)
                nn_cfg = VALIDATION_CONFIG["neural_network"]
                model.fit(
                    X_train_nn,
                    {"spread_pred": y_train_reg, "ext_prob": y_train_cls.astype(np.float32)},
                    epochs=int(nn_cfg["epochs"]), batch_size=int(nn_cfg["batch_size"]), verbose=0,
                )
                pred_rate, pred_probability = model.predict(X_holdout_nn, verbose=0)
                pred_cls = (np.asarray(pred_probability).reshape(-1) >= 0.5).astype(int)
                metric_rows.append(_metric_row(
                    eval_id, seed, fold_no, "TensorFlow_Multitask_NN", n_train, n_holdout,
                    train_group_count, holdout_group_count, class_counts, has_both_classes,
                    y_holdout_reg, np.asarray(pred_rate).reshape(-1), y_holdout_cls, pred_cls,
                ))
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass

    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    metrics_df = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    details = {
        "status": "completed_exploratory_grouped_cv",
        "seed_list": seeds,
        "folds_per_seed": n_splits,
        "evaluation_folds": eval_id,
        "independent_experiment_groups": group_count,
        "recommended_minimum_groups": int(VALIDATION_CONFIG["recommended_minimum_independent_groups"]),
        "recommended_minimum_groups_passed": group_count >= int(VALIDATION_CONFIG["recommended_minimum_independent_groups"]),
        "models_evaluated": sorted(metrics_df["model_family"].unique().tolist()) if not metrics_df.empty else [],
        "tensorflow_status": "included" if use_tf else (tf_skip_reason or "not_requested"),
        "leakage_check": "All 25 train/holdout group-intersection assertions passed.",
        "caution": "The five repeated 5-fold runs are correlated evaluations, not 25 statistically independent experiments. Hyperparameter/model selection requires nested grouped CV or a separately reserved group-level test set.",
    }
    if not metrics_df.empty:
        aggregate = metrics_df.groupby("model_family").agg(
            mae_mm_s_mean=("mae_mm_s", "mean"), mae_mm_s_std=("mae_mm_s", "std"),
            rmse_mm_s_mean=("rmse_mm_s", "mean"), rmse_mm_s_std=("rmse_mm_s", "std"),
            r2_mean=("r2_score", "mean"), r2_std=("r2_score", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        ).reset_index()
        details["aggregate_mean_std_descriptive_only"] = aggregate.replace({np.nan: None}).to_dict(orient="records")
    return manifest_df, metrics_df, details


def _metric_row(
    eval_id: int, seed: int, fold: int, model_name: str,
    n_train: int, n_holdout: int, n_train_groups: int, n_holdout_groups: int,
    class_counts: Counter, has_both_classes: bool,
    y_reg: np.ndarray, pred_reg: np.ndarray, y_cls: np.ndarray, pred_cls: np.ndarray,
) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_reg, pred_reg))
    rmse = float(np.sqrt(mean_squared_error(y_reg, pred_reg)))
    r2 = float(r2_score(y_reg, pred_reg)) if len(y_reg) >= 2 else float("nan")
    bal_acc = float(balanced_accuracy_score(y_cls, pred_cls))
    macro_f1 = float(f1_score(y_cls, pred_cls, average="macro", zero_division=0))
    note = ""
    if not has_both_classes:
        note = "Holdout contains only one class; balanced accuracy/F1 are not fully representative. Inspect group-level class distribution."
    return {
        "eval_id": eval_id, "seed": seed, "fold": fold, "model_family": model_name,
        "n_train_records": n_train, "n_holdout_records": n_holdout,
        "n_train_groups": n_train_groups, "n_holdout_groups": n_holdout_groups,
        "holdout_class_0": int(class_counts.get(0, 0)), "holdout_class_1": int(class_counts.get(1, 0)),
        "classification_holdout_has_both_classes": bool(has_both_classes),
        "mae_mm_s": mae, "rmse_mm_s": rmse, "r2_score": r2,
        "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "evaluation_note": note,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Curated empirical CSV with the documented schema.")
    parser.add_argument("--metadata", type=Path, default=None, help="Companion metadata JSON; defaults to <input stem>.metadata.json.")
    parser.add_argument("--allow-unverified-provenance", action="store_true", help="Run exploratory CV despite missing metadata declarations; outputs must not be used as validated scientific claims.")
    parser.add_argument("--include-tensorflow", action="store_true", help="Optionally compare a TensorFlow multitask network, only when there are at least the configured number of independent experiment groups and TensorFlow is installed.")
    args = parser.parse_args(argv)

    data_path = args.input.resolve()
    metadata_path = args.metadata.resolve() if args.metadata else data_path.with_suffix(".metadata.json")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = _base_report(data_path, metadata_path)

    if not data_path.exists():
        report["audit_status"] = "blocked_missing_input"
        report["model_evaluation_status"] = "not_run"
        report["blockers"].append(
            f"Input CSV does not exist: {data_path}. Do not substitute the bundled five-row BASS-II/FLEX excerpts or create synthetic rows to fill the file."
        )
        _empty_result_csvs()
        _write_json(REPORTS_DIR / "data_audit_report.json", report)
        print("BLOCKED: curated empirical dataset is missing. Audit JSON written; manifest/metrics contain headers only.")
        print(f"Expected CSV: {data_path}")
        print(f"Audit report: {REPORTS_DIR / 'data_audit_report.json'}")
        return 2

    report["data_sha256"] = _sha256(data_path)
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        report["audit_status"] = "blocked_unreadable_input"
        report["blockers"].append(f"Could not read CSV: {type(exc).__name__}: {exc}")
        _empty_result_csvs()
        _write_json(REPORTS_DIR / "data_audit_report.json", report)
        print(f"BLOCKED: {report['blockers'][-1]}")
        return 2

    report["dataset_name"] = "Empirical combustion dataset as identified in companion metadata, if provided"
    report["table_shape"] = {"records": int(len(df)), "columns": int(len(df.columns))}
    report["columns"] = df.columns.astype(str).tolist()
    report["duplicate_full_rows"] = int(df.duplicated().sum())
    if "record_id" in df.columns:
        report["duplicate_record_id_count"] = int(df["record_id"].astype(str).duplicated().sum())
        if report["duplicate_record_id_count"]:
            report["warnings"].append("record_id is not unique; ensure records can be traced unambiguously to source rows.")
    report["feature_audit"] = _feature_stats(df, FEATURE_COLS)
    report["target_audit"] = _feature_stats(df, [TARGET_REG, TARGET_CLS])
    report["source_distribution"] = (
        {str(k): int(v) for k, v in df["dataset_source"].fillna("<missing>").value_counts(dropna=False).items()}
        if "dataset_source" in df else {}
    )
    report["experiment_group_count"] = int(df["experiment_group"].nunique(dropna=True)) if "experiment_group" in df else 0
    report["rows_per_group"] = (
        {str(k): int(v) for k, v in df.groupby("experiment_group", dropna=False).size().items()}
        if "experiment_group" in df else {}
    )
    report["required_columns"] = REQUIRED_COLUMNS
    report["missing_required_columns"] = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    structural_blockers: list[str] = []
    if report["missing_required_columns"]:
        structural_blockers.append("Required modeling columns are missing: " + ", ".join(report["missing_required_columns"]))
    if len(df) < 1:
        structural_blockers.append("Input CSV has zero rows.")
    if "experiment_group" in df and df["experiment_group"].isna().any():
        structural_blockers.append("experiment_group contains missing values; every row must map to its physical experiment/run.")
    if "experiment_group" in df and df["experiment_group"].astype(str).str.strip().eq("").any():
        structural_blockers.append("experiment_group contains blank values.")
    if "dataset_source" in df and df["dataset_source"].isna().any():
        structural_blockers.append("dataset_source contains missing values.")

    for col in [*FEATURE_COLS, TARGET_REG, TARGET_CLS]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if col in (TARGET_REG, TARGET_CLS) and numeric.isna().any():
                structural_blockers.append(f"Target column {col!r} has missing/non-numeric values; labels will not be silently invented or dropped.")
            elif col in FEATURE_COLS and numeric.notna().sum() == 0:
                structural_blockers.append(f"Feature column {col!r} has no numeric values.")
            non_missing_values = numeric.dropna().to_numpy(dtype=float)
            if len(non_missing_values) and (~np.isfinite(non_missing_values)).any():
                structural_blockers.append(f"Column {col!r} contains +/-inf values; clean them explicitly before model evaluation.")

    if TARGET_CLS in df.columns:
        cls_values = set(pd.to_numeric(df[TARGET_CLS], errors="coerce").dropna().unique().tolist())
        if not cls_values.issubset({0, 1}):
            structural_blockers.append(f"{TARGET_CLS} must be encoded as 0/1 after an explicitly documented mapping; observed values: {sorted(map(str, cls_values))}.")
        elif len(cls_values) < 2:
            structural_blockers.append(f"{TARGET_CLS} contains only one class; classification evaluation is not meaningful.")

    n_groups = int(df["experiment_group"].nunique(dropna=True)) if "experiment_group" in df else 0
    required_groups = int(VALIDATION_CONFIG["required_minimum_groups_for_five_fold_cv"])
    if n_groups < required_groups:
        structural_blockers.append(f"Only {n_groups} independent groups were found; five-fold GroupKFold requires at least {required_groups} groups.")
    if n_groups < int(VALIDATION_CONFIG["recommended_minimum_independent_groups"]):
        report["warnings"].append(
            f"Only {n_groups} independent groups were found; below the configured recommendation of {VALIDATION_CONFIG['recommended_minimum_independent_groups']}. Any performance estimate will have substantial uncertainty."
        )

    # Flag exact predictor+target duplicates that appear under multiple group IDs.
    modeling_cols = [c for c in [*FEATURE_COLS, TARGET_REG, TARGET_CLS] if c in df.columns]
    if modeling_cols and "experiment_group" in df.columns and all(c in df.columns for c in modeling_cols):
        tmp = df[modeling_cols + ["experiment_group"]].copy()
        try:
            cross_group_duplicate_rows = int(
                tmp.groupby(modeling_cols, dropna=False)["experiment_group"].nunique().gt(1).sum()
            )
        except TypeError:
            cross_group_duplicate_rows = None
        report["exact_feature_target_patterns_in_multiple_groups"] = cross_group_duplicate_rows
        if cross_group_duplicate_rows:
            report["warnings"].append(
                "At least one exact feature+target pattern occurs across distinct group IDs. Inspect whether this is a legitimate repeated condition or accidental row duplication; group separation alone cannot fix mislabeled group IDs."
            )

    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            structural_blockers.append(f"Companion metadata JSON is unreadable: {type(exc).__name__}: {exc}")
    sources = sorted(df["dataset_source"].dropna().astype(str).unique().tolist()) if "dataset_source" in df else []
    meta_blockers = _metadata_blockers(metadata, sources)
    report["metadata_review"] = {
        "file_exists": metadata_path.exists(),
        "metadata_blockers": meta_blockers,
        "meaning": "Metadata assertions are declared by the project team and still require a scientifically responsible human review; they are not automatically proven by this script.",
    }

    all_blockers = structural_blockers + ([] if args.allow_unverified_provenance else meta_blockers)
    report["structural_blockers"] = structural_blockers
    report["blockers"] = all_blockers
    report["provenance_and_target_leakage"].update({
        "metadata_declaration_present": metadata is not None,
        "metadata_review_required": True,
        "no_synthetic_rule_leakage": "not_provable_automatically",
        "provenance_gate_overridden_for_exploration": bool(args.allow_unverified_provenance and meta_blockers),
    })

    if all_blockers:
        report["audit_status"] = "completed_with_blockers"
        report["model_evaluation_status"] = "blocked_scientific_readiness_checks_failed"
        _empty_result_csvs()
        _write_json(REPORTS_DIR / "data_audit_report.json", report)
        print("BLOCKED: scientific data/metadata checks did not pass. No cross-validation metrics were generated.")
        for blocker in all_blockers:
            print(f" - {blocker}")
        print(f"Audit report: {REPORTS_DIR / 'data_audit_report.json'}")
        return 2

    report["audit_status"] = "completed"
    if args.allow_unverified_provenance and meta_blockers:
        report["warnings"].append("Exploratory mode: provenance/target review blockers were overridden. Do not present resulting metrics as scientifically validated.")

    try:
        manifest_df, metrics_df, cv_details = _run_cv(df, include_tensorflow=args.include_tensorflow)
    except Exception as exc:
        report["model_evaluation_status"] = "failed_runtime_error"
        report["runtime_error"] = f"{type(exc).__name__}: {exc}"
        _empty_result_csvs()
        _write_json(REPORTS_DIR / "data_audit_report.json", report)
        print(f"FAILED: grouped CV error: {type(exc).__name__}: {exc}")
        return 3

    manifest_path = REPORTS_DIR / "experiment_split_manifest.csv"
    metrics_path = REPORTS_DIR / "cv_fold_metrics.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    report["model_evaluation_status"] = cv_details["status"]
    report["cross_validation"] = cv_details
    report["output_files"] = {
        "audit_report": str(REPORTS_DIR / "data_audit_report.json"),
        "split_manifest": str(manifest_path),
        "fold_metrics": str(metrics_path),
    }
    _write_json(REPORTS_DIR / "data_audit_report.json", report)
    print(f"Audit status: {report['audit_status']}")
    print(f"CV folds evaluated: {cv_details['evaluation_folds']}")
    print(f"Manifest rows: {len(manifest_df)}; metric rows: {len(metrics_df)}")
    if cv_details.get("aggregate_mean_std_descriptive_only"):
        print(pd.DataFrame(cv_details["aggregate_mean_std_descriptive_only"]).to_string(index=False))
    print("Reminder: repeated CV folds are correlated; do not treat 25 folds as 25 independent experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
