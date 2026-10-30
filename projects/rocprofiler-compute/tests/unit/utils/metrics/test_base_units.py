# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/base_units.py."""

import re
from typing import Set

import numpy as np
import pytest
import yaml

from config import analysis_configs_dir
from utils.metrics.base_units import (
    SCALED_UNITS,
    base_unit,
    from_base_unit_value,
    to_base_unit_value,
)

# A config unit with a scale prefix and a rate, such as GB/s or GFLOPs, or KB per Wave.
SCALED_UNIT_PATTERN = re.compile(r"^(K|[GMT])(B|b|FLOP|IOP)")


def config_units() -> Set[str]:
    """Every metric unit string in the shipped analysis configs."""
    units: Set[str] = set()

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "unit" and isinstance(value, str):
                    units.add(value)
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    for config_file in analysis_configs_dir.rglob("*.yaml"):
        collect(yaml.safe_load(config_file.read_text(encoding="utf-8")))
    return units


@pytest.mark.parametrize(
    ("config_unit", "stored_unit"),
    [
        ("GB/s", "Bytes/s"),
        ("Gbps", "Bytes/s"),
        ("GFLOP/s", "FLOP/s"),
        ("GFLOPs", "FLOP/s"),
        ("GIOP/s", "IOP/s"),
        ("GIOPs", "IOP/s"),
        ("KB per Wave", "Bytes per Wave"),
    ],
)
def test_base_unit_names_the_base_unit_of_a_scaled_config_unit(
    config_unit, stored_unit
):
    """Each scaled config unit is stored under its base unit name."""
    assert base_unit(config_unit) == stored_unit


@pytest.mark.parametrize(
    "unit",
    ["Bytes/s", "Nanoseconds", "FLOPs/Byte", "Percent", "Cycles", "Requests/s", None],
)
def test_base_unit_keeps_units_that_are_already_base(unit):
    """Units without a scaled form are stored as they are configured."""
    assert base_unit(unit) == unit


def test_to_base_unit_value_multiplies_a_scaled_value_by_1e9():
    """A GB/s value is per ns, so the stored Bytes/s value is 1e9 times larger."""
    assert to_base_unit_value("GB/s", "Value", 1.5) == 1.5e9
    assert to_base_unit_value("GFLOP/s", "Peak (Empirical)", 81715.2) == 81715.2 * 1e9
    assert to_base_unit_value("Gbps", "Avg", 0.25) == 0.25e9


def test_to_base_unit_value_multiplies_kb_per_wave_by_1024():
    """KB per wave is stored as Bytes per Wave, so the value is 1024 times larger."""
    assert to_base_unit_value("KB per Wave", "Value", 2.5) == 2.5 * 1024


def test_to_base_unit_value_keeps_percent_of_peak_unscaled():
    """A percent is a ratio of two values in the same unit, so its scale stays."""
    assert to_base_unit_value("GB/s", "Percent of Peak", 42.5) == 42.5


def test_to_base_unit_value_keeps_base_unit_values_unchanged():
    """Values whose unit is already a base unit are stored exactly as computed."""
    assert to_base_unit_value("Bytes/s", "Value", 1 / 3) == 1 / 3
    assert to_base_unit_value("Nanoseconds", "Avg", 12.5) == 12.5


@pytest.mark.parametrize("value", [None, "N/A", "text"])
def test_to_base_unit_value_passes_non_numbers_through(value):
    """Missing values and text stay as they are, so NULL is never stored as 0."""
    assert to_base_unit_value("GB/s", "Value", value) is value


def test_to_base_unit_value_keeps_nan_as_nan():
    """NaN stays NaN after scaling, so it still stores as NULL."""
    assert np.isnan(to_base_unit_value("GB/s", "Value", float("nan")))


def test_to_base_unit_value_accepts_numpy_numbers():
    """Values read from pandas frames are numpy scalars and scale like floats."""
    assert to_base_unit_value("GIOP/s", "Value", np.float64(2.0)) == 2.0e9


def test_every_scaled_config_unit_has_a_base_unit_rule():
    """A scaled unit added to a config must also get a rule here.

    Otherwise its values would be stored per ns under a unit name that
    claims a base unit.
    """
    scaled_in_configs = {
        unit for unit in config_units() if SCALED_UNIT_PATTERN.match(unit)
    }
    assert scaled_in_configs, "expected scaled units in the shipped configs"
    assert scaled_in_configs <= set(SCALED_UNITS)


@pytest.mark.parametrize(
    ("unit", "value_name", "config_value"),
    [
        ("GB/s", "Value", 1.5),
        ("GFLOP/s", "Peak (Empirical)", 81715.2),
        ("KB per Wave", "Avg", 2.5),
        ("Bytes/s", "Value", 1 / 3),
        ("Nanoseconds", "Avg", 12.5),
    ],
)
def test_from_base_unit_value_inverts_to_base_unit_value(
    unit, value_name, config_value
):
    """Reading a stored value back in its config unit recovers the value."""
    stored = to_base_unit_value(unit, value_name, config_value)
    assert from_base_unit_value(unit, value_name, stored) == pytest.approx(
        config_value, rel=1e-12
    )


def test_from_base_unit_value_divides_a_stored_value_by_its_scale():
    """A Bytes/s value stored for a GB/s metric is shown in GB/s again."""
    assert from_base_unit_value("GB/s", "Value", 1.5e9) == 1.5
    assert from_base_unit_value("KB per Wave", "Value", 2560.0) == 2.5


def test_from_base_unit_value_keeps_percent_of_peak_unscaled():
    """Percent of Peak keeps its scale in both directions."""
    assert from_base_unit_value("GB/s", "Percent of Peak", 42.5) == 42.5


@pytest.mark.parametrize("value", [None, "N/A"])
def test_from_base_unit_value_passes_non_numbers_through(value):
    """A NULL or text value is returned as it is, never shown as 0."""
    assert from_base_unit_value("GB/s", "Value", value) is value


def test_from_base_unit_value_keeps_nan_as_nan():
    """NaN stays NaN, so a value that was never calculated is still NULL."""
    assert np.isnan(from_base_unit_value("GB/s", "Value", float("nan")))
