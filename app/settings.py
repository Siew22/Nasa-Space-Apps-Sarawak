from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def find_project_root() -> Path:
    """
    通过双特征锚点定位项目根目录。
    必须同时存在 config/simulation_model.json 与 requirements.txt，
    若遍历至系统根路径仍未定位，坚决抛出 RuntimeError，拒绝隐式猜测。
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config" / "simulation_model.json").exists() and (parent / "requirements.txt").exists():
            return parent

    raise RuntimeError(
        "无法定位项目根目录！请确认代码运行于项目工作区内，且根目录下包含 "
        "'config/simulation_model.json' 与 'requirements.txt'。"
    )


def allowed_origins() -> list[str]:
    origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:8501,http://127.0.0.1:8501")
    return [o.strip() for o in origins.split(",") if o.strip()]


def read_json_config() -> dict[str, Any]:
    cfg_path = ROOT / "config" / "simulation_model.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_ai_config() -> dict[str, Any]:
    cfg_path = ROOT / "config" / "ai_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"AI 配置文件缺失: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)
