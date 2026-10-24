from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "dashboard_records.csv"
DEFAULT_OUT = ROOT / "reports"


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "dataset"


def repeated_group_splits(
    frame: pd.DataFrame,
    group_col: str,
    repeats: int = 5,
    n_splits: int = 5,
    validation_size: float = 0.2,
    min_groups: int = 10,
) -> tuple[pd.DataFrame, dict]:
    data = frame.copy().reset_index(drop=True)
    if "record_id" not in data.columns:
        data["record_id"] = [f"row:{i}" for i in range(len(data))]
    data[group_col] = data[group_col].astype("string")
    data = data[data[group_col].notna() & data[group_col].ne("")].reset_index(drop=True)
    n_groups = data[group_col].nunique()

    if n_groups < max(n_splits + 2, min_groups):
        raise ValueError(f"只有 {n_groups} 个独立 group，不足以执行严谨的交叉验证，至少需要 {max(n_splits + 2, min_groups)} 个。")

    groups = data[group_col].to_numpy()
    ids = data["record_id"].astype(str).to_numpy()
    output_rows = []
    checks = []

    for repeat_idx, seed in enumerate(range(repeats), start=1):
        outer = GroupKFold(n_splits=n_splits)
        for fold_idx, (trainval_idx, test_idx) in enumerate(outer.split(data, groups=groups), start=1):
            trainval_groups = groups[trainval_idx]
            inner = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed * 1000 + fold_idx)
            local_train, local_val = next(inner.split(trainval_idx, groups=trainval_groups))

            train_idx = trainval_idx[local_train]
            val_idx = trainval_idx[local_val]

            train_g = set(groups[train_idx])
            val_g = set(groups[val_idx])
            test_g = set(groups[test_idx])

            overlap_tv = len(train_g & val_g)
            overlap_tt = len(train_g & test_g)
            overlap_vt = len(val_g & test_g)

            if overlap_tv or overlap_tt or overlap_vt:
                raise AssertionError("检测到组泄漏 (Group Leakage)！")

            for part, idxs in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
                for i in idxs:
                    output_rows.append({
                        "record_id": ids[i],
                        "repeat": repeat_idx,
                        "fold": fold_idx,
                        "partition": part,
                        "group_id": str(groups[i]),
                    })

            checks.append({
                "repeat": repeat_idx,
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "overlap_clean": bool(overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0),
            })

    report = {
        "rows": len(data),
        "distinct_groups": n_groups,
        "repeats": repeats,
        "folds": n_splits,
        "all_train_validation_test_group_overlaps_zero": True,
        "fold_details": checks,
    }
    return pd.DataFrame(output_rows), report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-groups", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = df[df["dataset_key"].astype(str) == args.dataset_key].copy()
    manifest, report = repeated_group_splits(df, group_col=args.group, repeats=args.repeats, n_splits=args.folds, min_groups=args.min_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = slug(args.dataset_key)
    manifest.to_csv(args.output_dir / f"split_manifest_{stem}.csv", index=False)
    with (args.output_dir / f"split_audit_{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Group 交叉验证切分审计完毕，完全无数据泄漏: {args.output_dir / f'split_audit_{stem}.json'}")


if __name__ == "__main__":
    main()