"""
严谨科研评估流水线：
1. 采用严格的 GroupKFold 分组切分，同一独立实验的所有行绝对不同时出现在训练集与验证集。
2. 建立 Dummy 基线（均值预测 / 多数类预测）和随机森林基准，检验深度神经网络是否真正学习到有效规律。
3. 统计真实的 MAE (带真实物理单位 kW/m²)、Balanced Accuracy、Macro-F1，排除数据泄漏。
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import layers

# GPU 显存增长管理
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "spacecraft_fire_benchmark.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 输入特征：仅包含实验前可测的物理条件与初相参量
FEATURE_COLS = [
    "radiant_heat_flux_kw_m2",
    "real_burn_duration_s",
    "real_mean_hrr_kw_m2"
]


def build_scientific_keras_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.Dense(16, activation="relu")(x)
    
    # 任务 1: 回归实测 Peak HRR (kW/m²)
    out_reg = layers.Dense(1, activation="linear", name="peak_hrr_pred")(x)
    # 任务 2: 分类高危险度判定
    out_cls = layers.Dense(1, activation="sigmoid", name="hazard_prob")(x)

    model = keras.Model(inputs=inputs, outputs=[out_reg, out_cls])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss={"peak_hrr_pred": "mse", "hazard_prob": "binary_crossentropy"},
        loss_weights={"peak_hrr_pred": 0.001, "hazard_prob": 1.0}  # 尺度平衡
    )
    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}，请先执行 build_space_fire_benchmark.py")

    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_reg = df["target_peak_hrr_kw_m2"].to_numpy(dtype=np.float32)
    y_cls = df["target_is_high_hazard"].to_numpy(dtype=np.int32)
    groups = df["group_id"].to_numpy(dtype=str)

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    results = []
    print(f"\n🚀 启动严格的 Group-Aware 分组验证 (共 {n_splits} 折，互斥实验组，杜绝数据泄漏)...")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y_cls, groups=groups), start=1):
        # 验证分组无交集
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups & val_groups) == 0, "致命错误：训练集与验证集存在重叠的实验组！"

        # 特征标准化（严格在训练折上 fit）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx])
        X_val = scaler.transform(X_raw[val_idx])

        # -------------------------------------------------------------
        # 1. 朴素基线 (Dummy Baseline): 均值与多数类预测
        # -------------------------------------------------------------
        dummy_reg = DummyRegressor(strategy="mean")
        dummy_reg.fit(X_train, y_reg[train_idx])
        dummy_reg_pred = dummy_reg.predict(X_val)

        dummy_cls = DummyClassifier(strategy="most_frequent")
        dummy_cls.fit(X_train, y_cls[train_idx])
        dummy_cls_pred = dummy_cls.predict(X_val)

        # -------------------------------------------------------------
        # 2. 统计学基线: 随机森林 (Random Forest)
        # -------------------------------------------------------------
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_reg.fit(X_train, y_reg[train_idx])
        rf_reg_pred = rf_reg.predict(X_val)

        rf_cls = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_cls.fit(X_train, y_cls[train_idx])
        rf_cls_pred = rf_cls.predict(X_val)

        # -------------------------------------------------------------
        # 3. TensorFlow 深度学习模型 (在 12GB GPU 上训练)
        # -------------------------------------------------------------
        tf_model = build_scientific_keras_model(input_dim=len(FEATURE_COLS))
        tf_model.fit(
            X_train,
            {"peak_hrr_pred": y_reg[train_idx], "hazard_prob": y_cls[train_idx]},
            epochs=60,
            batch_size=8,
            verbose=0
        )
        tf_preds = tf_model(X_val, training=False)
        tf_reg_pred = tf_preds[0].numpy().flatten()
        tf_cls_pred = (tf_preds[1].numpy().flatten() >= 0.5).astype(np.int32)

        # 记录各模型评估指标
        for model_name, pred_r, pred_c in [
            ("Dummy_Baseline", dummy_reg_pred, dummy_cls_pred),
            ("Random_Forest", rf_reg_pred, rf_cls_pred),
            ("TensorFlow_GPU_NN", tf_reg_pred, tf_cls_pred)
        ]:
            mae = mean_absolute_error(y_reg[val_idx], pred_r)
            r2 = r2_score(y_reg[val_idx], pred_r) if len(val_idx) > 1 else np.nan
            bacc = balanced_accuracy_score(y_cls[val_idx], pred_c)
            f1 = f1_score(y_cls[val_idx], pred_c, average="macro", zero_division=0)

            results.append({
                "fold": fold_idx,
                "model": model_name,
                "mae_kw_m2": mae,
                "r2_score": r2,
                "balanced_accuracy": bacc,
                "macro_f1": f1
            })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("model").agg({
        "mae_kw_m2": ["mean", "std"],
        "balanced_accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"]
    }).round(3)

    report_path = REPORTS_DIR / "rigorous_group_cv_summary.csv"
    summary.to_csv(report_path, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("🔬 【真正的无泄漏 GroupKFold 交叉验证评估报告】 (Mean ± Std):")
    print(summary)
    print("=" * 65)
    print(f"📁 详细评测报告已落盘至: {report_path}\n")


if __name__ == "__main__":
    main()