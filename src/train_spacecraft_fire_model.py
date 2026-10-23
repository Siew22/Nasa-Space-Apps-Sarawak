"""Train an exploratory peak heat-release regressor on measured burn summaries.

The three inputs are measured during or after a burn test. High hazard follows
the existing project peak-HRR threshold; it is not a second trained classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "spacecraft_fire_benchmark.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
MODEL_STEM = "spacecraft_fire_peak_hrr_v1"

FEATURE_COLS = [
    "radiant_heat_flux_kw_m2",
    "real_burn_duration_s",
    "real_mean_hrr_kw_m2",
]
TARGET_COL = "target_peak_hrr_kw_m2"
GROUP_COL = "group_id"
HAZARD_THRESHOLD_KW_M2 = 350.0
MIN_ROWS = 20
MIN_GROUPS = 5


def load_and_validate_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str, np.ndarray]:
    """Reject missing, placeholder, and invalid supervised training data."""
    if not path.is_file():
        raise ValueError(f"Training CSV not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read training CSV {path}: {exc}") from exc

    required = [*FEATURE_COLS, TARGET_COL, GROUP_COL]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {', '.join(missing)}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"At least {MIN_ROWS} measured rows are required; found {len(df)}.")

    numeric = df[[*FEATURE_COLS, TARGET_COL]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Features and measured peak HRR must be finite numbers with no blanks.")
    if (numeric[FEATURE_COLS[0]] < 0).any() or (numeric[FEATURE_COLS[1:]] <= 0).any().any():
        raise ValueError("Heat flux must be nonnegative; burn duration and mean HRR must be positive.")
    if (numeric[TARGET_COL] <= 0).any():
        raise ValueError("Measured peak HRR must be positive.")
    if (numeric[TARGET_COL] + 1e-6 < numeric[FEATURE_COLS[2]]).any():
        raise ValueError("Measured peak HRR cannot be lower than mean HRR for the same burn.")

    group_ids = df[GROUP_COL].astype("string").str.strip()
    if group_ids.isna().any() or group_ids.eq("").any():
        raise ValueError("Every row needs a nonempty independent experiment group_id.")
    if group_ids.nunique() < MIN_GROUPS:
        raise ValueError(f"At least {MIN_GROUPS} independent experiment groups are required.")

    for column in FEATURE_COLS:
        if numeric[column].nunique() < 2:
            raise ValueError(f"{column} is constant; this looks like placeholder data.")
    if numeric[FEATURE_COLS].drop_duplicates().shape[0] < 10:
        raise ValueError("Fewer than 10 distinct input combinations; training would be misleading.")
    if numeric[TARGET_COL].nunique() < 3:
        raise ValueError("Measured peak HRR has fewer than three distinct values.")

    if "target_is_high_hazard" in df:
        supplied = pd.to_numeric(df["target_is_high_hazard"], errors="coerce")
        derived = (numeric[TARGET_COL] >= HAZARD_THRESHOLD_KW_M2).astype(int)
        if supplied.isna().any() or not supplied.isin([0, 1]).all():
            raise ValueError("target_is_high_hazard must contain only 0 or 1 if present.")
        if not np.array_equal(supplied.to_numpy(dtype=int), derived.to_numpy(dtype=int)):
            raise ValueError("target_is_high_hazard disagrees with the 350 kW/m2 peak-HRR rule.")
    if "dataset_source" in df:
        sources = df["dataset_source"].astype("string").str.lower().fillna("")
        if sources.str.contains(r"placeholder|synthetic|fabricated|fallback", regex=True).any():
            raise ValueError("One or more rows are marked as placeholder or synthetic.")

    # A laboratory holdout is more demanding than a file holdout when multiple
    # experiments come from the same lab. Otherwise use the experiment IDs.
    group_column = GROUP_COL
    groups = group_ids
    if "laboratory" in df:
        laboratories = df["laboratory"].astype("string").str.strip()
        if laboratories.notna().all() and laboratories.ne("").all() and laboratories.nunique() >= MIN_GROUPS:
            group_column, groups = "laboratory", laboratories

    x = numeric[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = numeric[TARGET_COL].to_numpy(dtype=np.float32)
    group_array = groups.to_numpy(dtype=str)
    for train_idx, _ in GroupKFold(n_splits=min(5, len(set(group_array)))).split(x, y, group_array):
        if np.unique(y[train_idx]).size < 2:
            raise ValueError("A grouped training fold has constant peak HRR; add more varied independent experiments.")
    return df, x, y, group_column, group_array


def build_scientific_keras_model(tf, input_dim: int):
    """Train on standardized peak HRR; the saved wrapper returns kW/m2."""
    inputs = tf.keras.Input(shape=(input_dim,), name="scaled_burn_summary")
    hidden = tf.keras.layers.Dense(32, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
    output = tf.keras.layers.Dense(1, name="standardized_peak_hrr")(hidden)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss="mse")
    return model


def train_network(tf, x_train: np.ndarray, y_train: np.ndarray, *, epochs: int, batch_size: int, seed: int):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train))
    if y_scale <= 0:
        raise ValueError("Training fold has no variation in measured peak HRR.")
    model = build_scientific_keras_model(tf, x_train.shape[1])
    model.fit(
        x_train,
        (y_train - y_mean) / y_scale,
        epochs=epochs,
        batch_size=min(batch_size, len(x_train)),
        verbose=0,
    )
    return model, y_mean, y_scale


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    true_hazard = (y_true >= HAZARD_THRESHOLD_KW_M2).astype(int)
    predicted_hazard = (y_pred >= HAZARD_THRESHOLD_KW_M2).astype(int)
    both_classes = len(np.unique(true_hazard)) == 2
    return {
        "mae_kw_m2": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kw_m2": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(true_hazard, predicted_hazard)) if both_classes else None,
        "macro_f1": float(f1_score(true_hazard, predicted_hazard, average="macro", labels=[0, 1], zero_division=0)) if both_classes else None,
    }


def evaluate_grouped(
    tf, x_raw: np.ndarray, y: np.ndarray, groups: np.ndarray, *, epochs: int, batch_size: int
) -> tuple[pd.DataFrame, dict[str, dict[str, float | None]], int]:
    n_splits = min(5, len(set(groups)))
    names = ("Dummy_Baseline", "Random_Forest", "TensorFlow_NN")
    predictions = {name: np.full(len(y), np.nan, dtype=np.float64) for name in names}
    folds = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=n_splits).split(x_raw, y, groups), start=1):
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError("An experiment group appeared in both training and validation.")
        scaler = StandardScaler().fit(x_raw[train_idx])
        x_train, x_val = scaler.transform(x_raw[train_idx]), scaler.transform(x_raw[val_idx])
        y_train = y[train_idx]
        dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
        forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(x_train, y_train)
        network, y_mean, y_scale = train_network(
            tf, x_train, y_train, epochs=epochs, batch_size=batch_size, seed=42 + fold
        )
        current = {
            "Dummy_Baseline": np.maximum(dummy.predict(x_val), 0),
            "Random_Forest": np.maximum(forest.predict(x_val), 0),
            "TensorFlow_NN": np.maximum(network(x_val, training=False).numpy().reshape(-1) * y_scale + y_mean, 0),
        }
        for name, values in current.items():
            predictions[name][val_idx] = values
            folds.append({
                "fold": fold,
                "model": name,
                "train_groups": len(set(groups[train_idx])),
                "validation_groups": len(set(groups[val_idx])),
                **model_metrics(y[val_idx], values),
            })
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Grouped validation did not predict every row.")
    return pd.DataFrame(folds), {name: model_metrics(y, values) for name, values in predictions.items()}, n_splits


def _scope(df: pd.DataFrame) -> str:
    materials = df["material"].dropna().astype(str).str.upper().unique().tolist() if "material" in df else []
    sources = df["dataset_source"].dropna().astype(str).unique().tolist() if "dataset_source" in df else []
    if materials == ["PMMA"] and any("macfp" in source.lower() for source in sources):
        return "Post-test PMMA cone-calorimeter measurements; spaceflight performance unverified."
    return "Post-test burn-summary data supplied for training; broader applicability unverified."


def _companion_metadata(path: Path, rows: int) -> dict:
    sidecar = path.with_name(f"{path.stem.removesuffix('_experiments')}_metadata.json")
    if not sidecar.is_file():
        return {}
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read companion data metadata {sidecar}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Companion data metadata must be a JSON object: {sidecar}")
    if "included_experiments" in metadata and metadata["included_experiments"] != rows:
        raise ValueError("Companion metadata row count disagrees with the training CSV.")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV of measured experiments; defaults to the repository benchmark")
    parser.add_argument("--epochs", type=int, default=60, help="Fixed neural-network epochs per fold and final fit (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8, help="Neural-network batch size (default: 8)")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")

    data_path = args.data.expanduser().resolve()
    try:
        df, x_raw, y, group_column, groups = load_and_validate_data(data_path)
        source_metadata = _companion_metadata(data_path, len(df))
    except ValueError as exc:
        parser.error(str(exc))

    import tensorflow as tf

    print(f"Training device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"Data: {data_path} | {len(df)} rows | {len(set(groups))} {group_column} groups")
    print("Checking independent groups against mean-prediction and random-forest baselines...")
    fold_df, summary, n_splits = evaluate_grouped(
        tf, x_raw, y, groups, epochs=args.epochs, batch_size=args.batch_size
    )
    summary_df = pd.DataFrame([
        {"model": name, "rows": len(y), "group_folds": n_splits, **metrics}
        for name, metrics in summary.items()
    ])

    scaler = StandardScaler().fit(x_raw)
    network, y_mean, y_scale = train_network(
        tf, scaler.transform(x_raw), y, epochs=args.epochs, batch_size=args.batch_size, seed=42
    )
    # The saved Keras model expects scaled inputs and directly outputs peak HRR
    # in physical units. Its output is never negative.
    peak_output = tf.keras.layers.Rescaling(scale=y_scale, offset=y_mean, name="peak_hrr_kw_m2")(network.output)
    peak_output = tf.keras.layers.ReLU(name="nonnegative_peak_hrr_kw_m2")(peak_output)
    inference_model = tf.keras.Model(network.input, peak_output, name=MODEL_STEM)

    sources = df["dataset_source"].dropna().astype(str).unique().tolist() if "dataset_source" in df else []
    materials = df["material"].dropna().astype(str).unique().tolist() if "material" in df else []
    commits = df["source_commit"].dropna().astype(str).unique().tolist() if "source_commit" in df else []
    try:
        display_path = data_path.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = data_path.name
    source_label = f"{display_path}; dataset_source={', '.join(sources) if sources else 'unspecified'}"
    if len(commits) == 1:
        source_label += f"; source_commit={commits[0]}"
    high_count = int(np.sum(y >= HAZARD_THRESHOLD_KW_M2))
    hazard_validated = 0 < high_count < len(y)
    metadata = {
        "format_version": 1,
        "model_file": f"{MODEL_STEM}.keras",
        "feature_columns": FEATURE_COLS,
        "feature_units": ["kW/m2", "s", "kW/m2"],
        "output": TARGET_COL,
        "output_units": "kW/m2",
        "input_stage": "after_burn_test",
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "hazard_threshold_kw_m2": HAZARD_THRESHOLD_KW_M2,
        "source": source_label,
        "scope": _scope(df),
        "source_preprocessing": source_metadata,
        "training_data": {
            "csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "rows": len(df),
            "independent_group_column": group_column,
            "independent_groups": len(set(groups)),
            "materials": materials,
            "dataset_sources": sources,
            "unique_input_rows": int(pd.DataFrame(x_raw).drop_duplicates().shape[0]),
            "feature_ranges": {
                column: {"min": float(np.min(x_raw[:, i])), "max": float(np.max(x_raw[:, i]))}
                for i, column in enumerate(FEATURE_COLS)
            },
            "peak_hrr_min_kw_m2": float(np.min(y)),
            "peak_hrr_max_kw_m2": float(np.max(y)),
            "hazard_class_counts": {
                "below_threshold": len(y) - high_count,
                "at_or_above_threshold": high_count,
            },
        },
        "validation": {
            "method": f"{n_splits}-fold GroupKFold by {group_column}; scaler fitted inside each training fold",
            "metrics": summary,
            "hazard_discrimination_validated": hazard_validated,
            "tensorflow_beats_dummy_mae": summary["TensorFlow_NN"]["mae_kw_m2"] < summary["Dummy_Baseline"]["mae_kw_m2"],
            "tensorflow_beats_random_forest_mae": summary["TensorFlow_NN"]["mae_kw_m2"] < summary["Random_Forest"]["mae_kw_m2"],
        },
        "readiness": "experimental; not validated for spaceflight or unseen materials",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_STEM}.keras"
    metadata_path = MODELS_DIR / f"{MODEL_STEM}.metadata.json"
    staged_model = MODELS_DIR / f".{MODEL_STEM}.staging.keras"
    staged_metadata = MODELS_DIR / f".{MODEL_STEM}.staging.json"
    try:
        inference_model.save(staged_model)
        staged_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staged_model, model_path)
        os.replace(staged_metadata, metadata_path)
    finally:
        staged_model.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)

    summary_path = REPORTS_DIR / "rigorous_group_cv_summary.csv"
    folds_path = REPORTS_DIR / "rigorous_group_cv_folds.csv"
    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved experimental model: {model_path}")
    print(f"Saved scaler and validation metadata: {metadata_path}")
    print(f"Grouped validation reports: {summary_path} and {folds_path}")
    if not hazard_validated:
        print("WARNING: No measured examples on both sides of 350 kW/m2; high-hazard discrimination is unvalidated.")
    if not metadata["validation"]["tensorflow_beats_dummy_mae"]:
        print("WARNING: TensorFlow did not beat mean prediction in grouped validation.")
    print("The model is exploratory. Input rows need the same after-test feature units and order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
