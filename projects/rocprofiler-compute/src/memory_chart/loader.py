# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load memory chart layout JSON files by architecture name."""

import json
from pathlib import Path
from typing import Any

_LAYOUTS_DIR = Path(__file__).resolve().parent / "layouts"

_ARCH_TO_FILE: dict[str, str] = {
    "gfx1150": "gfx115x.json",
    "gfx1151": "gfx115x.json",
    "gfx115x": "gfx115x.json",
    "gfx1250": "gfx1250.json",
    "gfx908": "gfx90x.json",
    "gfx90a": "gfx90x.json",
    "gfx940": "gfx94x.json",
    "gfx941": "gfx94x.json",
    "gfx942": "gfx94x.json",
    "gfx950": "gfx950.json",
}


def load_layout(arch: str) -> dict[str, Any]:
    """Load the memory chart layout JSON for the given arch."""
    filename = _ARCH_TO_FILE.get(arch)
    if filename is None:
        raise ValueError(f"Unknown architecture {arch!r}. Known: {list(_ARCH_TO_FILE)}")
    path = _LAYOUTS_DIR / filename
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def list_architectures() -> list[str]:
    """Return all supported architecture names."""
    return list(_ARCH_TO_FILE)


def list_layout_files() -> list[Path]:
    """Return paths to all layout JSON files (excluding manifest)."""
    return sorted(p for p in _LAYOUTS_DIR.glob("*.json") if p.name != "manifest.json")
