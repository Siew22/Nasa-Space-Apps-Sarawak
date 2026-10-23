from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def allowed_origins() -> list[str]:
    origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:8501,http://127.0.0.1:8501")
    return [o.strip() for o in origins.split(",") if o.strip()]


def read_json_config() -> dict[str, Any]:
    cfg_path = ROOT / "config" / "simulation_model.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)
