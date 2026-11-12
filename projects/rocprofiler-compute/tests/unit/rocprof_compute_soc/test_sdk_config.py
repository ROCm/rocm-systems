# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Offline Navi48 checks that the generic tests do not cover.

Every counter the gfx1201 panels use must be defined for gfx1201.
Passing this test does not establish SDK exposure or hardware measurement
units; those require validation on the supported ROCm stack.
"""

from typing import Dict, Set

import pytest
import yaml

import config
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils.utils_counter_defs import parse_counters_text

ARCH = "gfx1201"
SDK_CONFIG = (
    config.rocprof_compute_home
    / "rocprof_compute_soc"
    / "profile_configs"
    / "sdk_config.yaml"
)
PANEL_DIR = config.analysis_configs_dir / ARCH


@pytest.fixture(scope="module")
def gfx1201_definitions() -> Dict[str, Dict]:
    """Return each counter's gfx1201 definition."""
    sdk = yaml.safe_load(SDK_CONFIG.read_text(encoding="utf-8"))
    return {
        counter["name"]: definition
        for counter in sdk["rocprofiler-sdk"]["counters"]
        for definition in counter["definitions"]
        if ARCH in definition["architectures"]
    }


@pytest.fixture(scope="module")
def panel_counters() -> Set[str]:
    """Return the counters used by gfx1201 analysis metric expressions."""
    counters: Set[str] = set()
    for metric in iter_analysis_config_metrics(PANEL_DIR, set()):
        for expression in metric.expressions:
            counters.update(parse_counters_text(expression)[0])
    return counters


def test_gfx1201_panel_counters_are_defined(panel_counters, gfx1201_definitions):
    assert not panel_counters - gfx1201_definitions.keys()
