"""CLI helpers for `perfxpert config show / set`."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from perfxpert.config._config import PerfXpertConfig, load_config


_VALID_FIELDS = set(PerfXpertConfig.model_fields.keys())


def _config_path() -> Path:
    return Path(os.environ.get("HOME", str(Path.home()))) / ".config" / "perfxpert" / "config.yaml"


def _read_yaml() -> Dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text()) or {}
    return parsed if isinstance(parsed, dict) else {}


def _write_yaml(data: Dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True))


def _coerce(field: str, raw: str) -> Any:
    hint = PerfXpertConfig.model_fields[field].annotation
    hint_name = str(hint)
    if "bool" in hint_name:
        return raw.lower() in ("1", "true", "yes", "on")
    if "int" in hint_name:
        return int(raw)
    if "float" in hint_name:
        return float(raw)
    return raw


def run_config_show() -> None:
    cfg = load_config()
    dumped = yaml.safe_dump(cfg.model_dump(), sort_keys=True)
    sys.stdout.write(dumped)


def run_config_set(field: str, raw_value: str) -> None:
    if field not in _VALID_FIELDS:
        valid = ", ".join(sorted(_VALID_FIELDS))
        sys.stderr.write(f"error: unknown field {field!r}; valid: {valid}\n")
        raise SystemExit(2)
    value = _coerce(field, raw_value)
    data = _read_yaml()
    data[field] = value
    # Validate that the resulting config is loadable
    PerfXpertConfig(**{**data})
    _write_yaml(data)
    sys.stdout.write(f"set {field}={value}\n")


__all__ = ["run_config_show", "run_config_set"]
