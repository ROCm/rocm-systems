"""Structured knowledge base — YAML files validated against JSON schemas.

Tools load knowledge via `from perfxpert.knowledge import load_yaml`.
Schemas are validated at import time in development (via test suite).
"""

from pathlib import Path
from typing import Any

import yaml

_KNOWLEDGE_DIR = Path(__file__).parent


def load_yaml(name: str) -> Any:
    """Load a knowledge YAML file by stem name.

    Example:
        from perfxpert.knowledge import load_yaml
        specs = load_yaml("gpu_specs")
        mi300x = specs["gfx942"]

    Args:
        name: Filename stem (no .yaml extension)

    Returns:
        Parsed YAML content (dict, list, or scalar depending on file).

    Raises:
        FileNotFoundError: if the YAML file doesn't exist.
    """
    path = _KNOWLEDGE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {path}")
    return yaml.safe_load(path.read_text())
