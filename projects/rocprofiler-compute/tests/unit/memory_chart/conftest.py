# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared fixtures and helpers for memory chart tests."""

import json
from pathlib import Path

import common
import pytest
import yaml

LAYOUTS_DIR = Path(common.SRC) / "memory_chart" / "layouts"
ANALYSIS_CONFIGS_DIR = Path(common.SRC) / "rocprof_compute_soc" / "analysis_configs"
MEMORY_CHART_YAML = "0300_memory_chart.yaml"

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"

LAYOUT_FILES = sorted(
    p for p in LAYOUTS_DIR.glob("*.json") if p.name != "manifest.json"
)

LAYOUT_IDS = [p.stem for p in LAYOUT_FILES]


@pytest.fixture(params=LAYOUT_FILES, ids=LAYOUT_IDS)
def layout_path(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture yielding each layout JSON path."""
    return request.param


@pytest.fixture(params=LAYOUT_FILES, ids=LAYOUT_IDS)
def layout_data(request: pytest.FixtureRequest) -> dict:
    """Parametrized fixture yielding parsed layout data."""
    with request.param.open(encoding="utf-8") as fobj:
        return json.load(fobj)


def load_layout_json(name: str) -> dict:
    """Load a named layout file."""
    path = LAYOUTS_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as fobj:
        return json.load(fobj)


def panel_yaml_path(arch: str) -> Path:
    """Return the memory chart YAML config path for an arch."""
    return ANALYSIS_CONFIGS_DIR / arch / MEMORY_CHART_YAML


def panel_yaml_metric_keys(arch: str) -> frozenset[str]:
    """Extract metric names from an arch's memory chart YAML."""
    config_path = panel_yaml_path(arch)
    if not config_path.exists():
        return frozenset()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return frozenset(
        metric_name
        for data_source in config["Panel Config"]["data source"]
        for metric_name in data_source["metric_table"]["metric"]
    )


def yaml_metric_names(arch: str) -> set[str]:
    """Extract all metric names from an arch's memory chart YAML."""
    return set(panel_yaml_metric_keys(arch))


def sample_metrics_for_arch(arch: str) -> dict[str, float]:
    """Build a sample metric dict with all keys set to 1.0."""
    return dict.fromkeys(panel_yaml_metric_keys(arch), 1.0)
