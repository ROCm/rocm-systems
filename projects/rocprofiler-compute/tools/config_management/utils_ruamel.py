# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from typing import Any, Union

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# --- Round-trip YAML (for writing) ---
RT_YAML = YAML(typ="rt")
RT_YAML.preserve_quotes = True
RT_YAML.width = 4096  # prevent unwanted line wrapping
RT_YAML.indent(mapping=2, sequence=2, offset=0)
RT_YAML.explicit_start = False
RT_YAML.explicit_end = False

# --- Read-only YAML (safe loader) ---
RO_YAML = YAML(typ="safe")
RO_YAML.width = 4096


def load_yaml(
    filepath: Union[str, Path],
    *,
    round_trip: bool = False,
) -> Any:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    yaml = RT_YAML if round_trip else RO_YAML

    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f) or CommentedMap()


def save_yaml(data: Any, filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        RT_YAML.dump(data, f)
