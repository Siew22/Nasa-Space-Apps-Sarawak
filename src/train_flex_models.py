"""Train and export the two deployed FLEX models (microgravity droplet combustion, ISS).

  flex_droplet_burning_rate_constant -> Ridge regression      -> models/flex_burning_rate_ridge_v1.json
  flex_extinction_outcome            -> logistic regression   -> models/flex_extinction_logistic_v1.json

Why these two: they are the best-scoring families in the leakage-audited cross-validation (reports/cv_summary.json)
and the ones the README reports. The recipe is the CV recipe, imported from src/run_cv_evaluation.py (median
imputation, standard scaling, Ridge / class-balanced logistic regression, the same small tuning grid).

Final setting: the grid value with the best mean score (MAE / log loss, the CV inner-selection criteria) over
group-disjoint 5-fold splits of ALL rows, averaged over the 5 protocol seeds. It is then refit on all rows.
The performance to quote is the nested CV result in reports/cv_summary.json, which evaluated this whole procedure
on unseen test days; it is copied into each model file together with the CV report's timestamp.

The models are linear, so they are exported as plain numbers (imputation medians, scaling, coefficients) instead of
a pickle: exact, readable, and the dashboard can show each input's contribution.

Run:  python -m src.train_flex_models
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import log_loss, mean_absolute_error

from src.group_cv import randomized_group_kfold
from src.run_cv_evaluation import MODEL_GRIDS, _target_column, load_task_frame, make_model

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "target_registry.json"
CV_SUMMARY_PATH = ROOT / "reports" / "cv_summary.json"
CV_PROTOCOL_PATH = ROOT / "reports" / "cv_protocol.json"
MODELS_DIR = ROOT / "models"
DEPLOY = {  # task_id -> (model family, baseline family, output file)
    "flex_droplet_burning_rate_constant": ("ridge", "mean_baseline", "flex_burning_rate_ridge_v1.json"),
    "flex_extinction_outcome": ("logistic_regression", "majority_baseline", "flex_extinction_logistic_v1.json"),
}
OUTER_FOLDS = 5


def task_rows(task: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Rows, inputs, target and groups exactly as src/run_cv_evaluation.py keeps them."""
    feats = [f["column"] for f in task["features"]]
    target = _target_column(task)
    df = load_task_frame(task)
    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = np.isfinite(y_all) & ((y_all > 0) if task["task_type"] == "regression" else y_all.isin([0, 1]))
    df = df[keep].reset_index(drop=True)
    x = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return df, x, df[target].to_numpy(float), df[task["group_column"]].astype(str).to_numpy()


def select_setting(family: str, task_type: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   seeds: list[int]) -> tuple[dict, list[dict]]:
    """Grid value with the best mean grouped-CV score (MAE or log loss) over all seeds and folds."""
    scored = []
    for params in MODEL_GRIDS[family]:
        scores = []
        for seed in seeds:
            for k, test_idx in enumerate(randomized_group_kfold(groups, OUTER_FOLDS, seed)):
                train = np.setdiff1d(np.arange(len(y)), test_idx)
                model = make_model(family, params, seed * 100 + k).fit(x[train], y[train])
                if task_type == "regression":
                    scores.append(mean_absolute_error(y[test_idx], model.predict(x[test_idx])))
                else:
                    p = model.predict_proba(x[test_idx])[:, list(model.classes_).index(1)]
                    scores.append(log_loss(y[test_idx].astype(int), np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        scored.append({"params": params, "mean_score": float(np.mean(scores)), "folds": len(scores)})
    best = min(scored, key=lambda s: s["mean_score"])
    return best["params"], scored


def export(task: dict, family: str, baseline: str, seeds: list[int], cv_summary: dict) -> dict:
    df, x, y, groups = task_rows(task)
    ttype = task["task_type"]
    params, scored = select_setting(family, ttype, x, y, groups, seeds)
    pipe = make_model(family, params, seeds[0]).fit(x, y)
    imputer, scaler, model = pipe.named_steps["impute"], pipe.named_steps["scale"], pipe.named_steps["model"]
    coef = np.ravel(model.coef_)
    intercept = float(np.ravel([model.intercept_])[0])
    if ttype == "classification":
        assert list(model.classes_) == [0, 1], model.classes_
    cv_task = next(t for t in cv_summary["tasks"] if t["task_id"] == task["task_id"])
    data_path = ROOT / task["data_file"]
    features = task["features"]
    return {
        "format_version": 1,
        "task_id": task["task_id"],
        "title": task["title"], "title_zh": task.get("title_zh", task["title"]),
        "model_family": family,
        "task_type": ttype,
        "formula": ("prediction = intercept + sum(coef * (x_imputed - scaler_mean) / scaler_scale)" if ttype == "regression"
                    else "score = 1 / (1 + exp(-(intercept + sum(coef * (x_imputed - scaler_mean) / scaler_scale))))"),
        "feature_columns": [f["column"] for f in features],
        "feature_units": {f["column"]: f["unit"] for f in features},
        "imputer_median": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "output": (task["target"]["column"] if ttype == "regression" else task["label"]["column"]),
        "output_unit": task["target"]["unit"] if ttype == "regression" else "score for class 1 (flame went out)",
        "decision_threshold": task.get("decision_threshold"),
        "score_is_calibrated_probability": False if ttype == "classification" else None,
        "score_note": (None if ttype == "regression" else
                       "Trained with class_weight='balanced' (as in CV), which shifts scores toward 0.5. The score ranks "
                       "risk and the 0.5 threshold is the evaluated decision rule; it is not a calibrated probability."),
        "hyperparameters": params,
        "hyperparameter_selection": {
            "method": f"mean {'MAE' if ttype == 'regression' else 'log loss'} over group-disjoint {OUTER_FOLDS}-fold "
                      f"splits of all rows, seeds {seeds} (src/group_cv.py randomized_group_kfold)",
            "candidates": scored,
        },
        "training_data": {
            "file": task["data_file"],
            "csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "row_filter_column": task.get("row_filter_column"),
            "rows": int(len(y)),
            "independent_group_column": task["group_column"],
            "independent_groups": int(len(set(groups))),
            "feature_ranges": {c: {"min": float(np.nanmin(x[:, i])), "max": float(np.nanmax(x[:, i]))}
                               for i, c in enumerate(f["column"] for f in features)},
            **({"target_mean": float(y.mean()), "target_min": float(y.min()), "target_max": float(y.max())}
               if ttype == "regression" else {"class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())}}),
        },
        "cross_validation": {
            "source": "reports/cv_summary.json",
            "generated_at_utc": cv_summary.get("generated_at_utc"),
            "protocol": "5 seeds x 5 outer folds of whole test days; tuning on inner validation only",
            "model": cv_task["models"][family]["fold_level"],
            "baseline_family": baseline,
            "baseline": cv_task["models"][baseline]["fold_level"],
        },
        "scope": "Microgravity droplet combustion on the ISS (NASA FLEX, PSI-69): n-heptane or methanol droplets "
                 "in the tested pressure and gas-mixture ranges only.",
        "scope_zh": "国际空间站上的微重力液滴燃烧（NASA FLEX，PSI-69）：仅适用于正庚烷或甲醇液滴，以及已测试的压力与气体混合范围。",
        "trained_with": {"scikit_learn": sklearn.__version__, "script": "src/train_flex_models.py"},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    cv_summary = json.loads(CV_SUMMARY_PATH.read_text(encoding="utf-8"))
    seeds = json.loads(CV_PROTOCOL_PATH.read_text(encoding="utf-8"))["seeds"]
    tasks = {t["task_id"]: t for t in registry["tasks"]}
    MODELS_DIR.mkdir(exist_ok=True)
    for task_id, (family, baseline, file_name) in DEPLOY.items():
        meta = export(tasks[task_id], family, baseline, seeds, cv_summary)
        (MODELS_DIR / file_name).write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                            encoding="utf-8")
        chosen = ", ".join(f"{k}={v}" for k, v in meta["hyperparameters"].items())
        print(f"{task_id}: {family} ({chosen}) on {meta['training_data']['rows']} rows -> models/{file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
