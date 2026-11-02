# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit display rules and value formatting for the memory chart."""

import pytest

from memory_chart.units import PLAIN, display_unit, format_value, panel_units


@pytest.mark.parametrize(
    ("unit", "value", "expected"),
    [
        ("Percent", 66.66, "66.7%"),
        ("Bytes/s", 1.5e9, "1.500 GB/s"),
        ("Bytes/s", 0, "0.000 GB/s"),
        ("KB per Wave", 0.625, "0.625 KB"),
        ("Bytes per Workgroup", 32768, "32.0 KB"),
        ("Cycles", 68.24, "68.2 cycles"),
        ("Registers", 64, "64.0"),
        ("(Requests + $normUnit)", 119, "119"),
        ("(Instructions + $normUnit)", 12345, "1.23e+04"),
        (None, 3, "3.0"),
    ],
)
def test_value_is_formatted_by_its_unit(unit, value, expected):
    assert format_value(value, display_unit(unit) or PLAIN) == expected


@pytest.mark.parametrize("unit", ["Percent", "(Requests + $normUnit)"])
@pytest.mark.parametrize("value", [None, float("nan"), "N/A"])
def test_missing_value_shows_na(unit, value):
    assert format_value(value, display_unit(unit)) == "N/A"


@pytest.mark.parametrize(
    ("unit", "has_rule"),
    [
        ("Percent", True),
        ("(Wavefronts + $normUnit)", True),
        ("Furlongs", False),
        (None, False),
    ],
)
def test_only_config_units_have_a_display_rule(unit, has_rule):
    assert (display_unit(unit) is not None) is has_rule


def test_panel_units_skips_metrics_without_a_unit():
    table = {
        "metric": {"Hit": {"value": "x", "unit": "Percent"}, "Bare": {"value": "y"}}
    }
    assert panel_units({"data source": [{"metric_table": table}]}) == {"Hit": "Percent"}
