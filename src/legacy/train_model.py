from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- WSL2 12GB VRAM 显存自适应分配配置 ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 [CUDA] 成功启用 GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ GPU 配置警告: {e}")
else:
    print("ℹ️ 未检测到 GPU，使用 CPU 运算。")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_SAVE_PATH = ROOT / "models" / "microgravity_fire_mlp_v1.keras"
MATERIALS = ["PMMA", "SIBAL_FABRIC", "NOMEX", "DELRIN"]


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从下载的数据集整合特征"""
    bass_file = RAW_DIR / "PSI-25_BASS_II_experimental_table.csv"
    if not bass_file.exists():
        raise FileNotFoundError(f"找不到 BASS 文件，请先运行 python -m src.generate_synthetic_demo_tables（注意：生成的是合成演示数据，并非 NASA 实测）")

    df_bass = pd.read_csv(bass_file)
    rows_x = []
    y_spread = []
    y_extinct = []

    for _, r in df_bass.iterrows():
        g = 0.0 / 9.81  # 微重力
        o2 = (float(r["Calibrated initial O2 % by vol"]) - 21.0) / 20.0
        p = (101.3 - 101.3) / 60.0
        flow = (float(r["Fan display"]) / 100.0)

        # 映射材料
        mat_str = str(r["Fuel Sample Material"]).upper()
        if "PMMA" in mat_str:
            mat = "PMMA"
        elif "SIBAL" in mat_str or "COTTON" in mat_str:
            mat = "SIBAL_FABRIC"
        elif "DELRIN" in mat_str:
            mat = "DELRIN"
        else:
            mat = "NOMEX"

        mat_vec = [1.0 if mat == m else 0.0 for m in MATERIALS]
        rows_x.append([g, o2, p, flow] + mat_vec)

        # 实测与推断目标
        # 若风速极低且氧气 <= 21%，自熄概率较高
        is_ext = 1.0 if flow < 0.3 and r["Calibrated initial O2 % by vol"] <= 21.0 else 0.0
        spread = 0.0 if is_ext else round(0.3 + flow * 1.8 + (o2 * 0.8), 3)
        y_spread.append(spread)
        y_extinct.append(is_ext)

    return (
        np.array(rows_x, dtype=np.float32),
        np.array(y_spread, dtype=np.float32),
        np.array(y_extinct, dtype=np.float32),
    )


def build_model(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="physics_inputs")
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.15)(x)

    spread_rate = layers.Dense(1, activation="relu", name="spread_rate")(x)
    extinction = layers.Dense(1, activation="sigmoid", name="extinction")(x)

    model = keras.Model(inputs=inputs, outputs=[spread_rate, extinction], name="SpacecraftFirePredictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss={"spread_rate": "mse", "extinction": "binary_crossentropy"},
        metrics={"spread_rate": ["mae"], "extinction": ["accuracy"]},
    )
    return model


def main():
    X, y_s, y_e = load_training_data()
    X_train, X_val, ys_train, ys_val, ye_train, ye_val = train_test_split(
        X, y_s, y_e, test_size=0.2, random_state=42
    )

    print(f"\n⚡ 开始在 GPU 上训练 TensorFlow 燃烧多任务预测模型...")
    model = build_model(X_train.shape[1])
    model.fit(
        X_train,
        {"spread_rate": ys_train, "extinction": ye_train},
        validation_data=(X_val, {"spread_rate": ys_val, "extinction": ye_val}),
        epochs=100,
        batch_size=8,
        verbose=1,
    )

    eval_results = model.evaluate(X_val, {"spread_rate": ys_val, "extinction": ye_val}, verbose=0)
    print("\n" + "=" * 50)
    print("🎯 验证集最终评估指标:")
    for name, val in zip(model.metrics_names, eval_results):
        print(f"  - {name}: {val:.4f}")
    print("=" * 50)

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    print(f"💾 权重文件已落盘保存: {MODEL_SAVE_PATH}\n")


if __name__ == "__main__":
    main()