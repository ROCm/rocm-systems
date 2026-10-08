# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for memory chart tests.

Everything is discovered from the layout files and the analysis configs, so a new
layout or architecture is covered without editing the tests.
"""

from pathlib import Path
from typing import Any

import common
import yaml

from memory_chart.loader import layout_files, load_layout
from utils.utils_common import panel_tables

ANALYSIS_CONFIGS_DIR = Path(common.SRC) / "rocprof_compute_soc" / "analysis_configs"
MEMORY_CHART_YAML = "0300_memory_chart.yaml"


# Analysis-config directories that have a memory chart panel
CONFIG_ARCHS = sorted(
    path.parent.name for path in ANALYSIS_CONFIGS_DIR.glob(f"*/{MEMORY_CHART_YAML}")
)

# Every (layout file, arch it serves) pair
LAYOUT_ARCHS = [
    (path, arch) for path in layout_files() for arch in load_layout(path).archs
]
LAYOUT_ARCH_IDS = [f"{path.stem}-{arch}" for path, arch in LAYOUT_ARCHS]


def panel_config(arch: str) -> dict[str, Any]:
    """The panel 300 config of an analysis-config arch."""
    text = (ANALYSIS_CONFIGS_DIR / arch / MEMORY_CHART_YAML).read_text(encoding="utf-8")
    return yaml.safe_load(text)["Panel Config"]


def panel_metric_names(arch: str) -> list[str]:
    """Panel 300 metric names of an arch, in config order, duplicates kept."""
    return [
        name
        for _, table in panel_tables(panel_config(arch))
        for name in table.get("metric") or {}
    ]
