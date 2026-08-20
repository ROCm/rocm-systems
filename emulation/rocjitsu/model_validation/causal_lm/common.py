#!/usr/bin/env python3
"""Shared constants and helpers for causal-LM rocjitsu smoke tests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "models.json"
ROCJITSU_ROOT = SCRIPT_DIR.parents[1]

PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 1
RTOL = 1e-3
ATOL = 1e-3
ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "*.md",
    "*.jinja",
]
IGNORE_PATTERNS = [
    "*.bin",
    "*.pth",
    "*.pt",
    "optimizer.pt",
    "training_args.bin",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


def load_models(path: Path = DEFAULT_MANIFEST) -> dict[str, str]:
    models = load_json(path)
    if not isinstance(models, dict):
        raise ValueError(f"model manifest must be an object: {path}")
    for key, model_id in models.items():
        if not isinstance(key, str) or not isinstance(model_id, str):
            raise ValueError(f"model manifest must map string keys to model IDs: {path}")
    return models


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def resolve_model(models: dict[str, str], key: str) -> dict[str, str]:
    try:
        model_id = models[key]
    except KeyError as exc:
        known = ", ".join(sorted(models))
        raise ValueError(f"unknown model '{key}'. Known models: {known}") from exc
    return {"key": key, "id": model_id, "local_dir": model_slug(model_id)}


def prepend_env_path(current: str | None, paths: Iterable[str | Path]) -> str:
    items = [str(path) for path in paths if str(path)]
    if current:
        items.append(current)
    return ":".join(items)
