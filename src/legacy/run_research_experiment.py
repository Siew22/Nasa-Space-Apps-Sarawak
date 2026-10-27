"""
微重力燃烧科研评估流水线：
1. 读取带有物理无量纲特征的清洗数据集 (research_curated_dataset.csv)
2. 执行严格的 Repeated Group-Aware 5-Fold Cross Validation (5 seeds x 5 folds = 25 轮隔离评估)
3. 对比两套模型：
   - Baseline: Scikit-Learn Random Forest
   - Proposed: TensorFlow 2.16 Keras 多任务物理约束神经网络 (GPU 加速)
4. 输出均值 ± 标准差科研指标汇总与物理特征重要性排名
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import layers

# 配置 WSL2 GPU 显存自适应生长 (防止一次性占满 12GB)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "research_curated_dataset.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 选定物理特征列与目标列
FEATURE_COLS = [
    "mod_reynolds_number",           # 雷诺数
    "mod_log_richardson",            # 理查森数（浮力 vs 强迫对流）
    "mod_o2_partial_pressure_kpa",   # 真实氧分压
    "mod_buoyancy_velocity_scale",   # 约化浮力速度
    "mod_damkohler_proxy"            # 达姆科勒动力学参数
]
TARGET_REG = "target_rate_mm_s"
TARGET_CLS = "target_is_extinguished"


def build_keras_multitask_model(input_dim: int) -> keras.Model:
    """构建多任务物理感知神经网络 (共享主干表征 + 回归/分类双头输出)"""
    inputs = keras.Input(shape=(input_dim,), name="physics_features")
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    
    # 任务 A: 蔓延速率预测 (ReLU 保证物理非负性)
    out_spread = layers.Dense(1, activation="relu", name="out_spread")(x)
    # 任务 B: 自熄灭概率判定 (Sigmoid 输出 0~1 概率)
    out_extinct = layers.Dense(1, activation="sigmoid", name="out_extinct")(x)

    model = keras.Model(inputs=inputs, outputs=[out_spread, out_extinct], name="Microgravity_PINN_MultiTask")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss={
            "out_spread": "mse",
            "out_extinct": "binary_crossentropy"
        },
        loss_weights={"out_spread": 1.0, "out_extinct": 0.8}
    )
    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据集：{DATA_PATH}，请先执行: python -m src.research_dataset_curation")

    df = pd.read_csv(DATA_PATH)
    print(f"🔬 载入科研数据: 共 {len(df)} 组观测，{df['experiment_group'].nunique()} 个独立试验组。")

    X_raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_spread = df["spread_or_burn_rate"].to_numpy(dtype=np.float32)
    y_extinct = df["is_extinguished"].to_numpy(dtype=np.int32)
    groups = df["experiment_group"].to_numpy(dtype=str)

    repeats = 5
    folds = 5
    benchmark_records = []

    print(f"\n🚀 开始执行 25 轮严谨 Group-Aware 交叉验证 (5 Seeds × 5 Folds)...")

    for seed in range(repeats):
        gkf = GroupKFold(n_splits=folds)
        # 为每一轮 repeat 设定确定的随机状态（通过置换打乱 group 顺序）
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(seed)
        shuffled_groups = rng.permutation(unique_groups)
        group_to_idx = {g: i for i, g in enumerate(shuffled_groups)}
        shuffled_group_indices = np.array([group_to_idx[g] for g in groups])

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_raw, groups=shuffled_group_indices), start=1):
            # 严格验证组间无交叉泄漏
            train_g = set(groups[train_idx])
            test_g = set(groups[test_idx])
            assert len(train_g & test_g) == 0, "致命错误：检测到测试集与训练集发生组泄漏！"

            # 特征标准化（仅在训练集上 fit，避免测试集统计量泄漏）
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_raw[train_idx])
            X_test = scaler.transform(X_raw[test_idx])

            # ---------------------------------------------------------
            # 1. Baseline 模型: Random Forest (回归 + 分类)
            # ---------------------------------------------------------
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=seed * 10 + fold_idx)
            rf_reg.fit(X_train, y_spread[train_idx])
            rf_spread_pred = rf_reg.predict(X_test)

            rf_clf = RandomForestClassifier(n_estimators=100, random_state=seed * 10 + fold_idx)
            rf_clf.fit(X_train, y_extinct[train_idx])
            rf_ext_pred = rf_clf.predict(X_test)

            benchmark_records.append({
                "model_family": "RandomForest_Baseline",
                "repeat": seed + 1,
                "fold": fold_idx,
                "reg_mae": mean_absolute_error(y_spread[test_idx], rf_spread_pred),
                "reg_rmse": np.sqrt(mean_squared_error(y_spread[test_idx], rf_spread_pred)),
                "reg_r2": r2_score(y_spread[test_idx], rf_spread_pred),
                "cls_balanced_acc": balanced_accuracy_score(y_extinct[test_idx], rf_ext_pred),
                "cls_macro_f1": f1_score(y_extinct[test_idx], rf_ext_pred, average="macro", zero_division=0),
            })

            # ---------------------------------------------------------
            # 2. 深度学习模型: TensorFlow 2.16 多任务物理网络 (GPU 训练)
            # ---------------------------------------------------------
            tf_model = build_keras_multitask_model(input_dim=len(FEATURE_COLS))
            tf_model.fit(
                X_train,
                {"out_spread": y_spread[train_idx], "out_extinct": y_extinct[train_idx]},
                epochs=60,
                batch_size=8,
                verbose=0
            )

            preds = tf_model(X_test, training=False)
            tf_spread_pred = preds[0].numpy().flatten()
            tf_ext_prob = preds[1].numpy().flatten()
            tf_ext_pred = (tf_ext_prob >= 0.5).astype(np.int32)

            benchmark_records.append({
                "model_family": "TensorFlow_Physics_NN",
                "repeat": seed + 1,
                "fold": fold_idx,
                "reg_mae": mean_absolute_error(y_spread[test_idx], tf_spread_pred),
                "reg_rmse": np.sqrt(mean_squared_error(y_spread[test_idx], tf_spread_pred)),
                "reg_r2": r2_score(y_spread[test_idx], tf_spread_pred),
                "cls_balanced_acc": balanced_accuracy_score(y_extinct[test_idx], tf_ext_pred),
                "cls_macro_f1": f1_score(y_extinct[test_idx], tf_ext_pred, average="macro", zero_division=0),
            })

    res_df = pd.DataFrame(benchmark_records)

    # 导出逐折详细表现表
    detailed_path = REPORTS_DIR / "research_cv_folds_detail.csv"
    res_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    # 聚合生成学术汇报专用的 (Mean ± Std) 指标表
    summary = (
        res_df.groupby("model_family")
        .agg({
            "reg_mae": ["mean", "std"],
            "reg_rmse": ["mean", "std"],
            "reg_r2": ["mean", "std"],
            "cls_balanced_acc": ["mean", "std"],
            "cls_macro_f1": ["mean", "std"],
        })
        .round(4)
    )
    summary_path = REPORTS_DIR / "research_benchmark_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # 3. 特征重要性分析 (Permutation Feature Importance)
    # ---------------------------------------------------------
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X_raw)
    final_rf = RandomForestRegressor(n_estimators=150, random_state=42)
    final_rf.fit(X_scaled, y_spread)
    
    perm_res = permutation_importance(final_rf, X_scaled, y_spread, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        "feature_name": FEATURE_COLS,
        "importance_mean": np.round(perm_res.importances_mean, 4),
        "importance_std": np.round(perm_res.importances_std, 4)
    }).sort_values(by="importance_mean", ascending=False)

    imp_path = REPORTS_DIR / "research_feature_importance.csv"
    importance_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    # 控制台打印最终科研报告
    print("\n" + "=" * 65)
    print("🏆 【科研交叉验证核心结果】 (25 轮外层隔离测试集均值与方差):")
    print(summary)
    print("=" * 65)
    print("\n🔬 【微重力燃烧特征重要性排名】 (Top Physical Features):")
    print(importance_df.to_string(index=False))
    print("=" * 65)
    print(f"\n💾 汇报成果文件已生成：")
    print(f"  1. 汇总表: {summary_path}")
    print(f"  2. 详细单折表: {detailed_path}")
    print(f"  3. 特征重要性排名表: {imp_path}\n")


if __name__ == "__main__":
    main()