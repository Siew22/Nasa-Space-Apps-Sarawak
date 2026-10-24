"""Leakage-audited, repeated group-aware cross-validation for every task in config/target_registry.json.

A task is evaluated only when
  * its dataset is VERIFIED_SOURCE in reports/data_provenance_audit.json (run src/provenance_audit.py first),
  * its target unit (regression) or label mapping (classification) is marked verified in the registry, and
  * it has at least `min_independent_groups` independent groups.
Otherwise it is recorded as NOT_EVALUATED with machine-readable reason codes; no scores are produced.

Protocol per evaluated task (see src/group_cv.py):
  5 seeds x 5 outer folds of whole groups. The outer test fold is fixed first. The remaining groups are
  split into inner-train and inner-validation groups; models with a tuning grid pick their setting on
  inner validation only (MAE for regression, log loss for classification). The chosen setting is refit on
  the full outer-training partition and scored on the outer test fold exactly once. Imputation and scaling
  live inside each model pipeline, so they are fitted on training rows only. The classification decision
  threshold is fixed in the registry (not tuned). Every partition, prediction and metric is saved.

The 25 fold scores are NOT 25 independent experiments: the 5 repeats reuse the same rows, so fold scores
are positively correlated and their spread understates real-world uncertainty.

Run:  python -m src.run_cv_evaluation            (add --no-tensorflow to skip the neural network)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.group_cv import InsufficientGroupsError, audit_splits, make_repeated_splits

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "target_registry.json"
PROVENANCE_PATH = ROOT / "reports" / "data_provenance_audit.json"
REPORTS = ROOT / "reports"
OUT = {
    "fold_metrics": REPORTS / "cv_fold_metrics.csv",
    "manifest": REPORTS / "experiment_split_manifest.csv",
    "predictions": REPORTS / "cv_predictions.csv",
    "data_audit": REPORTS / "data_audit_report.json",
    "summary": REPORTS / "cv_summary.json",
    "protocol": REPORTS / "cv_protocol.json",
}

# Small, documented grids, tuned on inner validation only. Baselines and the deployed network use one
# predefined configuration (the network is the exact recipe of src/train_spacecraft_fire_model.py).
MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "mean_baseline": [{}],
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (1, 2, 4)],
    "tf_nn_deployed_config": [{"epochs": 60, "batch_size": 8}],
    "majority_baseline": [{}],
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "random_forest_classifier": [{"min_samples_leaf": m} for m in (1, 2, 4)],
}
REGRESSION_METRICS = ("mae", "rmse", "r2")
CLASSIFICATION_METRICS = ("balanced_accuracy", "macro_f1", "roc_auc", "precision_pos", "recall_pos", "recall_neg")
DEPENDENCE_NOTE = ("The 25 outer folds come from 5 repeats over the same rows, so they are correlated re-splits, "
                   "not 25 independent experiments. Per-seed pooled scores are reported to show repeat-to-repeat "
                   "variation; neither spread is a confidence interval for new experiments.")


class TFNetworkRegressor:
    """The deployed MaCFP network (inputs -> 32 -> 16 -> 1) with training-only imputation and scaling."""

    def __init__(self, seed: int, epochs: int = 60, batch_size: int = 8):
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TFNetworkRegressor":
        import tensorflow as tf

        from src.train_spacecraft_fire_model import train_network

        self.imputer_ = SimpleImputer(strategy="median").fit(x)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(x))
        xs = self.scaler_.transform(self.imputer_.transform(x)).astype(np.float32)
        self.net_, self.y_mean_, self.y_scale_ = train_network(
            tf, xs, y.astype(np.float32), epochs=self.epochs, batch_size=self.batch_size, seed=self.seed)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(self.imputer_.transform(x)).astype(np.float32)
        return np.maximum(self.net_(xs, training=False).numpy().reshape(-1) * self.y_scale_ + self.y_mean_, 0.0)


def make_model(family: str, params: dict, seed: int):
    impute = ("impute", SimpleImputer(strategy="median"))
    scale = ("scale", StandardScaler())
    if family == "mean_baseline":
        return Pipeline([impute, ("model", DummyRegressor(strategy="mean"))])
    if family == "ridge":
        return Pipeline([impute, scale, ("model", Ridge(alpha=params["alpha"]))])
    if family == "random_forest":
        return Pipeline([impute, scale, ("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=params["min_samples_leaf"], random_state=seed, n_jobs=-1))])
    if family == "tf_nn_deployed_config":
        return TFNetworkRegressor(seed=seed, epochs=params["epochs"], batch_size=params["batch_size"])
    if family == "majority_baseline":
        return Pipeline([impute, ("model", DummyClassifier(strategy="prior"))])
    if family == "logistic_regression":
        return Pipeline([impute, scale, ("model", LogisticRegression(
            C=params["C"], class_weight="balanced", max_iter=2000, random_state=seed))])
    if family == "random_forest_classifier":
        return Pipeline([impute, scale, ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=params["min_samples_leaf"], class_weight="balanced",
            random_state=seed, n_jobs=-1))])
    raise ValueError(f"unknown model family {family}")


def fit_predict(family: str, params: dict, x_train: np.ndarray, y_train: np.ndarray,
                x_eval: np.ndarray, seed: int, task_type: str) -> np.ndarray:
    """The only place models see labels: training labels in, predictions (or P(class 1)) for x_eval out."""
    model = make_model(family, params, seed).fit(x_train, y_train)
    if task_type == "regression":
        return model.predict(x_eval)
    classes = list(model.classes_)
    proba = model.predict_proba(x_eval)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(x_eval))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"mae": float(mean_absolute_error(y_true, y_pred)),
                           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))), "r2": None, "metric_note": ""}
    if len(y_true) < 2:
        out["metric_note"] = "r2: fewer than 2 test rows"
    elif float(np.var(y_true)) == 0.0:
        out["metric_note"] = "r2: no target variation in test fold"
    else:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = y_true.astype(int)
    pred = (prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both else None,
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=[0, 1], zero_division=0)) if both else None,
        "roc_auc": float(roc_auc_score(y_true, prob)) if both else None,
        "precision_pos": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, pred, pos_label=1, zero_division=0)) if (y_true == 1).any() else None,
        "recall_neg": float(recall_score(y_true, pred, pos_label=0, zero_division=0)) if (y_true == 0).any() else None,
        "metric_note": "" if both else "test fold contains one class only; class-balance metrics not computed",
    }
    return out


def _stats(values: list[float | None]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def _target_column(task: dict) -> str | None:
    return task["target"]["column"] if task["task_type"] == "regression" else task.get("label", {}).get("column")


def eligibility(task: dict, provenance: dict, min_groups: int) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    status = provenance.get("datasets", {}).get(task["dataset_key"], {}).get("status", "UNVERIFIED_SOURCE")
    if status != "VERIFIED_SOURCE":
        reasons.append({"code": "DATASET_NOT_VERIFIED", "dataset_status": status})
    if task["task_type"] == "regression":
        target = task.get("target", {})
        if not target.get("column"):
            reasons.append({"code": "TARGET_NOT_MEASURED"})
        elif not target.get("unit_verified"):
            reasons.append({"code": "TARGET_UNIT_UNVERIFIED", "note": target.get("unit_note", "")})
    else:
        label = task.get("label", {})
        if not label.get("source_column") or not label.get("column"):
            reasons.append({"code": "LABEL_NOT_OBSERVED"})
        elif not label.get("mapping_verified"):
            reasons.append({"code": "LABEL_MAPPING_UNVERIFIED"})
    data_file = task.get("data_file")
    if not reasons and data_file:
        if not (ROOT / data_file).exists():
            reasons.append({"code": "DATA_FILE_MISSING", "path": data_file})
        else:
            groups = load_task_frame(task)[task["group_column"]].nunique()
            if groups < min_groups:
                reasons.append({"code": "INSUFFICIENT_GROUPS", "groups": int(groups), "required": min_groups})
    return reasons


def load_task_frame(task: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / task["data_file"])
    if task.get("row_filter_column"):
        df = df[df[task["row_filter_column"]].astype(bool)]
    return df.reset_index(drop=True)


def data_audit(task: dict, df: pd.DataFrame) -> dict[str, Any]:
    feats = [f["column"] for f in task["features"]]
    target = _target_column(task)
    y = pd.to_numeric(df[target], errors="coerce")
    per_group = df[task["group_column"]].value_counts()
    audit = {
        "rows": int(len(df)),
        "independent_groups": int(per_group.size),
        "group_column": task["group_column"],
        "rows_per_group_summary": _stats(per_group.astype(float).tolist()),
        "duplicate_record_ids": int(df[task["record_id_column"]].duplicated().sum()),
        "duplicate_feature_target_rows": int(df[feats + [target]].duplicated().sum()),
        "missing_values": {c: int(df[c].isna().sum()) for c in feats + [target]},
        "features": task["features"],
        "post_test_features": [f["column"] for f in task["features"] if f.get("available") == "after_test"],
        "prediction_stage": task.get("prediction_stage", "before_test"),
    }
    if task["task_type"] == "regression":
        audit["invalid_targets"] = int((~np.isfinite(y) | (y <= 0)).sum())
        audit["target_distribution"] = _stats(y.dropna().tolist())
    else:
        audit["class_counts"] = {str(int(k)): int(v) for k, v in y.value_counts().items()}
    return audit


def evaluate_task(task: dict, df: pd.DataFrame, protocol: dict, families: list[str]
                  ) -> tuple[dict, list[dict], list[dict], list[dict]]:
    ttype = task["task_type"]
    feats = [f["column"] for f in task["features"]]
    target = _target_column(task)
    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = np.isfinite(y_all) & ((y_all > 0) if ttype == "regression" else y_all.isin([0, 1]))
    dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    x = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[target].to_numpy(float)
    groups = df[task["group_column"]].astype(str).to_numpy()
    ids = df[task["record_id_column"]].astype(str).to_numpy()
    threshold = float(task.get("decision_threshold", 0.5))

    splits = make_repeated_splits(groups, protocol["seeds"], protocol["outer_folds"],
                                  protocol["inner_validation_fraction"])
    audit = audit_splits(splits, groups, ids, protocol["outer_folds"])
    if not audit.passed:
        raise RuntimeError(f"Split audit failed for {task['task_id']}: {audit.failures[:3]}")

    fold_rows, manifest, predictions = [], [], []
    for s in splits:
        for part, idx in (("inner_train", s.inner_train_idx), ("inner_validation", s.inner_val_idx),
                          ("outer_test", s.test_idx)):
            manifest += [{"task_id": task["task_id"], "seed": s.seed, "fold": s.fold, "record_id": ids[i],
                          "group_id": groups[i], "partition": part} for i in idx]
        part_info = {
            "n_inner_train_rows": len(s.inner_train_idx), "n_inner_train_groups": len(set(groups[s.inner_train_idx])),
            "n_inner_val_rows": len(s.inner_val_idx), "n_inner_val_groups": len(set(groups[s.inner_val_idx])),
            "n_test_rows": len(s.test_idx), "n_test_groups": len(set(groups[s.test_idx])),
            "y_test_min": float(y[s.test_idx].min()), "y_test_max": float(y[s.test_idx].max()),
            "y_test_mean": float(y[s.test_idx].mean()),
        }
        model_seed = s.seed * 100 + s.fold
        for family in families:
            grid = MODEL_GRIDS[family]
            chosen, inner_score = grid[0], None
            if len(grid) > 1:  # model selection sees inner partitions only
                scores = []
                for params in grid:
                    pred = fit_predict(family, params, x[s.inner_train_idx], y[s.inner_train_idx],
                                       x[s.inner_val_idx], model_seed, ttype)
                    yv = y[s.inner_val_idx]
                    scores.append(mean_absolute_error(yv, pred) if ttype == "regression"
                                  else log_loss(yv.astype(int), np.clip(pred, 1e-6, 1 - 1e-6), labels=[0, 1]))
                best = int(np.argmin(scores))
                chosen, inner_score = grid[best], float(scores[best])
            pred = fit_predict(family, chosen, x[s.outer_train_idx], y[s.outer_train_idx], x[s.test_idx],
                               model_seed, ttype)
            metrics = (regression_metrics(y[s.test_idx], pred) if ttype == "regression"
                       else classification_metrics(y[s.test_idx], pred, threshold))
            fold_rows.append({"task_id": task["task_id"], "task_type": ttype, "seed": s.seed, "fold": s.fold,
                              "model": family, "selected_params": json.dumps(chosen, sort_keys=True),
                              "inner_selection_score": inner_score, **part_info, **metrics,
                              "evaluation_status": "EVALUATED"})
            predictions += [{"task_id": task["task_id"], "seed": s.seed, "fold": s.fold, "model": family,
                             "record_id": ids[i], "group_id": groups[i], "y_true": float(y[i]),
                             "y_pred": float(p), "result_type": "MODEL_PREDICTION_OUT_OF_FOLD"}
                            for i, p in zip(s.test_idx, pred)]

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(predictions)
    metric_names = REGRESSION_METRICS if ttype == "regression" else CLASSIFICATION_METRICS
    models = {}
    for family in families:
        f = folds[folds["model"] == family]
        per_seed = []
        for seed, p in preds[preds["model"] == family].groupby("seed"):
            yt, yp = p["y_true"].to_numpy(), p["y_pred"].to_numpy()
            m = regression_metrics(yt, yp) if ttype == "regression" else classification_metrics(yt, yp, threshold)
            per_seed.append({"seed": int(seed), **m})
        models[family] = {
            "fold_level": {k: _stats(f[k].tolist()) for k in metric_names},
            "per_seed_pooled": per_seed,
            "per_seed_pooled_summary": {k: _stats([d[k] for d in per_seed]) for k in metric_names},
            "selected_params_counts": f["selected_params"].value_counts().to_dict(),
            "folds_with_metric_notes": int((f["metric_note"] != "").sum()),
        }
    warnings, warnings_zh = [], []  # same messages in English and Chinese for the bilingual dashboard
    if dropped:
        warnings.append(f"{dropped} rows with a missing or invalid target were excluded before splitting")
        warnings_zh.append(f"划分前排除了 {dropped} 行目标缺失或无效的数据")
    if ttype == "regression":
        base = models.get("mean_baseline", {}).get("fold_level", {}).get("mae", {}).get("mean")
        for family, m in models.items():
            if family != "mean_baseline" and base is not None and m["fold_level"]["mae"]["mean"] >= base:
                warnings.append(f"{family} does not beat the mean baseline on MAE")
                warnings_zh.append(f"{family} 的 MAE 没有优于均值基线")
    else:
        for family, m in models.items():
            if m["folds_with_metric_notes"]:
                warnings.append(f"{family}: {m['folds_with_metric_notes']} test folds had one class only")
                warnings_zh.append(f"{family}：有 {m['folds_with_metric_notes']} 个测试折只包含一个类别")
    hazard = task.get("derived_classification")
    hazard_result = None
    if hazard:
        classes = np.unique((y >= 350.0).astype(int))
        hazard_result = ({"name": hazard["name"], "evaluation_status": "NOT_EVALUATED",
                          "reason": f"every measured target is on one side of the threshold (classes present: {classes.tolist()})"}
                         if classes.size < 2 else {"name": hazard["name"], "evaluation_status": "SEE_FOLD_METRICS"})
    summary = {
        "task_id": task["task_id"], "evaluation_status": "EVALUATED", "task_type": ttype,
        "target": task.get("target"), "label": task.get("label"), "decision_threshold": threshold if ttype != "regression" else None,
        "rows_used": int(len(df)), "rows_excluded_before_split": dropped,
        "class_counts": ({str(int(k)): int(v) for k, v in pd.Series(y).value_counts().items()}
                         if ttype != "regression" else None),
        "independent_groups": int(len(set(groups))), "group_column": task["group_column"],
        "outer_folds_total": len(splits), "models": models,
        "split_audit": {"passed": audit.passed, "checks": audit.checks, "failures": audit.failures},
        "derived_classification": hazard_result, "warnings": warnings, "warnings_zh": warnings_zh,
        "dependence_note": DEPENDENCE_NOTE,
        "pooled_note": (None if ttype == "regression" else
                        "Pooled classification scores combine probabilities from separately fitted models, which are not "
                        "on a common scale; e.g. the constant baseline's pooled ROC-AUC falls below 0.5 for this reason. "
                        "Fold-level scores are the primary result."),
    }
    return summary, fold_rows, manifest, predictions


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


FOLD_COLUMNS = ["task_id", "task_type", "seed", "fold", "model", "selected_params", "inner_selection_score",
                "n_inner_train_rows", "n_inner_train_groups", "n_inner_val_rows", "n_inner_val_groups", "n_test_rows",
                "n_test_groups", "y_test_min", "y_test_max", "y_test_mean", *REGRESSION_METRICS,
                *CLASSIFICATION_METRICS, "metric_note", "evaluation_status"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-tensorflow", action="store_true", help="skip the TensorFlow network family")
    args = parser.parse_args(argv)
    if not PROVENANCE_PATH.exists():
        parser.error("reports/data_provenance_audit.json is missing; run python -m src.provenance_audit first")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    protocol = registry["protocol"]
    tf_version = None
    if not args.no_tensorflow:
        try:
            import tensorflow as tf  # noqa: F401
            tf_version = tf.__version__
        except ImportError:
            print("TensorFlow not importable; the network family will be skipped", file=sys.stderr)

    started = datetime.now(timezone.utc)
    task_summaries, all_folds, all_manifest, all_preds, data_audits = [], [], [], [], {}
    for task in registry["tasks"]:
        reasons = eligibility(task, provenance, protocol["min_independent_groups"])
        if reasons:
            task_summaries.append({"task_id": task["task_id"], "evaluation_status": "NOT_EVALUATED",
                                   "task_type": task["task_type"], "dataset_key": task["dataset_key"],
                                   "reasons": reasons, "target": task.get("target"), "label": task.get("label")})
            print(f"{task['task_id']:36s} NOT_EVALUATED  {[r['code'] for r in reasons]}")
            continue
        df = load_task_frame(task)
        data_audits[task["task_id"]] = {"dataset_key": task["dataset_key"],
                                        "dataset_status": provenance["datasets"][task["dataset_key"]]["status"],
                                        "data_file": task["data_file"], "data_sha256": _sha(ROOT / task["data_file"]),
                                        "row_filter_column": task.get("row_filter_column"),
                                        **data_audit(task, df)}
        families = [m for m in task["models"] if m != "tf_nn_deployed_config" or tf_version]
        try:
            summary, folds, manifest, preds = evaluate_task(task, df, protocol, families)
        except InsufficientGroupsError as exc:
            task_summaries.append({"task_id": task["task_id"], "evaluation_status": "NOT_EVALUATED",
                                   "task_type": task["task_type"], "dataset_key": task["dataset_key"],
                                   "reasons": [{"code": "INSUFFICIENT_GROUPS", "detail": str(exc)}]})
            continue
        summary["dataset_key"] = task["dataset_key"]
        task_summaries.append(summary)
        all_folds += folds
        all_manifest += manifest
        all_preds += preds
        key = "mae" if task["task_type"] == "regression" else "balanced_accuracy"
        best = {m: (round(v["fold_level"][key]["mean"], 3) if v["fold_level"][key]["mean"] is not None else None)
                for m, v in summary["models"].items()}
        print(f"{task['task_id']:36s} EVALUATED      fold-mean {key} {best}  split audit passed={summary['split_audit']['passed']}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_folds, columns=FOLD_COLUMNS).to_csv(OUT["fold_metrics"], index=False)
    pd.DataFrame(all_manifest, columns=["task_id", "seed", "fold", "record_id", "group_id", "partition"]).to_csv(
        OUT["manifest"], index=False)
    pd.DataFrame(all_preds, columns=["task_id", "seed", "fold", "model", "record_id", "group_id", "y_true",
                                     "y_pred", "result_type"]).to_csv(OUT["predictions"], index=False)
    finished = datetime.now(timezone.utc)
    common = {"generated_at_utc": finished.isoformat(timespec="seconds"), "repository_commit": _git_commit(),
              "generated_by": "src/run_cv_evaluation.py"}
    OUT["data_audit"].write_text(json.dumps({"report": "data_audit_report", **common,
                                             "provenance_report": PROVENANCE_PATH.relative_to(ROOT).as_posix(),
                                             "provenance_generated_at_utc": provenance.get("generated_at_utc"),
                                             "tasks": data_audits}, indent=2, default=str) + "\n", encoding="utf-8")
    OUT["summary"].write_text(json.dumps({"report": "cv_summary", **common, "tasks": task_summaries},
                                         indent=2, default=str) + "\n", encoding="utf-8")
    OUT["protocol"].write_text(json.dumps({
        "report": "cv_protocol", **common,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "command": "python -m src.run_cv_evaluation" + (" --no-tensorflow" if args.no_tensorflow else ""),
        "seeds": protocol["seeds"], "outer_folds_per_seed": protocol["outer_folds"],
        "inner_validation_fraction_target": protocol["inner_validation_fraction"],
        "min_independent_groups": protocol["min_independent_groups"],
        "splitter": "src/group_cv.py randomized_group_kfold (scikit-learn 1.5 GroupKFold cannot shuffle) + inner_group_split",
        "partition_rule": "whole groups only; outer test fixed first; inner validation drawn from outer-training groups",
        "selection": {"regression_metric": "MAE on inner validation", "classification_metric": "log loss on inner validation",
                      "grids": MODEL_GRIDS,
                      "untuned": "baselines and tf_nn_deployed_config use one predefined configuration",
                      "classification_threshold": "fixed per task in config/target_registry.json (decision_threshold); never tuned"},
        "preprocessing": "median imputation and standard scaling inside each model, fitted on the training partition only",
        "network": "src/train_spacecraft_fire_model.py build_scientific_keras_model: Dense 32 relu -> Dense 16 relu -> Dense 1; Adam lr 0.003; MSE on standardized target",
        "dependence_note": DEPENDENCE_NOTE,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "tensorflow": tf_version},
        "group_definitions": {t["task_id"]: t.get("group_definition") for t in registry["tasks"]},
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print("Wrote " + ", ".join(p.relative_to(ROOT).as_posix() for p in OUT.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
