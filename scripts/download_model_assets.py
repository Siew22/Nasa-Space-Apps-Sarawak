"""
显式模型资产下载工具 (Explicit Offline Asset Provisioning)
在有网络的环境下一键将开源模型镜像完整固化到本地目录。
下载完成后系统即可断网，进入纯离线运行状态。
"""
from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "models" / "base_models" / "Qwen2.5-1.5B-Instruct"
EMBED_MODEL_DIR = ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"


def download_assets():
    print("📥 开始下载并固化开源模型资产到本地...")
    
    # 1. 下载语言模型基座 (Qwen2.5-1.5B-Instruct, Apache 2.0 协议)
    if not BASE_MODEL_DIR.exists() or not any(BASE_MODEL_DIR.iterdir()):
        print(f"📦 下载基座模型 Qwen/Qwen2.5-1.5B-Instruct -> {BASE_MODEL_DIR}")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            local_dir=str(BASE_MODEL_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
    else:
        print(f"✅ 基座模型已存在于本地: {BASE_MODEL_DIR}")

    # 2. 下载文本嵌入模型 (all-MiniLM-L6-v2, Apache 2.0 协议)
    if not EMBED_MODEL_DIR.exists() or not any(EMBED_MODEL_DIR.iterdir()):
        print(f"📦 下载嵌入模型 all-MiniLM-L6-v2 -> {EMBED_MODEL_DIR}")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.save(str(EMBED_MODEL_DIR))
    else:
        print(f"✅ 嵌入模型已存在于本地: {EMBED_MODEL_DIR}")

    print("\n🎉 全部模型资产本地化完成！后续训练与推理可在完全断网环境下执行。\n")


if __name__ == "__main__":
    download_assets()
