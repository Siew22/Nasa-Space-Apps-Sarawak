from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "dashboard_records.csv"
DEFAULT_OUT = ROOT / "reports"
META_COLUMNS = {"record_id", "dataset_key", "dataset_name", "source_file", "source_url", "source_doi", "source_row_number"}


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-groups", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    data = data[data["dataset_key"].astype(str) == args.dataset_key].dropna(subset=[args.target, args.group])

    n_groups = data[args.group].nunique()
    if n_groups < args.min_groups:
        raise SystemExit(f"Group 数 ({n_groups}) 小于最小需求 ({args.min_groups})。")

    exclude = {v.strip() for v in args.exclude.split(",") if v.strip()}
    non_features = META_COLUMNS | {args.target, args.group} | exclude
    X = data[[c for c in data.columns if c not in non_features]].dropna(axis=1, how="all")
    y = data[args.target]
    groups = data[args.group].astype(str).to_numpy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    results = []
    for r in range(args.repeats):
        gkf = GroupKFold(n_splits=args.folds)
        for fold, (trainval_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            model = RandomForestClassifier(n_estimators=100, random_state=r) if args.task == "classification" else RandomForestRegressor(n_estimators=100, random_state=r)
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X.iloc[trainval_idx], y.iloc[trainval_idx])
            preds = pipe.predict(X.iloc[test_idx])

            if args.task == "classification":
                results.append({
                    "repeat": r + 1, "fold": fold,
                    "accuracy": accuracy_score(y.iloc[test_idx], preds),
                    "macro_f1": f1_score(y.iloc[test_idx], preds, average="macro", zero_division=0)
                })
            else:
                results.append({
                    "repeat": r + 1, "fold": fold,
                    "mae": mean_absolute_error(y.iloc[test_idx], preds),
                    "r2": r2_score(y.iloc[test_idx], preds)
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    stem = f"{slug(args.dataset_key)}_{slug(args.target)}"
    res_df.to_csv(args.output_dir / f"cv_metrics_{stem}.csv", index=False)
    summary = res_df.mean().to_dict()
    print(f"📊 交叉验证基线评估完成 (均值): {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()