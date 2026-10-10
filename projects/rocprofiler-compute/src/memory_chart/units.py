# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""How the memory chart displays each panel 300 metric unit.

Units come from the metric config, never from a layout. A test checks that
every unit the shipped configs use has a rule here.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

UnitKind = Literal["percent", "bandwidth", "count", "value"]


@dataclass(frozen=True)
class DisplayUnit:
    """How to display values of one config unit."""

    kind: UnitKind
    suffix: str = ""
    precision: int = 1
    scale: float = 1.0


# For values whose unit has no display rule
PLAIN = DisplayUnit("value")
PERCENT = DisplayUnit("percent", "%")

_DISPLAY_UNITS: dict[str, DisplayUnit] = {
    "Percent": PERCENT,
    "Bytes/s": DisplayUnit("bandwidth", " GB/s", 3, 1e-9),
    "KB per Wave": DisplayUnit("value", " KB", 3),
    "Bytes per Workgroup": DisplayUnit("value", " KB", 1, 1 / 1024),
    "Cycles": DisplayUnit("value", " cycles"),
    "Registers": DisplayUnit("value"),
    "Workgroups per CU": DisplayUnit("value"),
    "Workgroups per WGP": DisplayUnit("value"),
}

# Per-normalization counts, e.g. "(Requests + $normUnit)"
_COUNT_UNIT = re.compile(r"^\(\w+ \+ \$normUnit\)$")
_COUNT = DisplayUnit("count")


def display_unit(unit: Optional[str]) -> Optional[DisplayUnit]:
    """The display rule for *unit*, or None if it has none."""
    if unit in _DISPLAY_UNITS:
        return _DISPLAY_UNITS[unit]
    if unit and _COUNT_UNIT.match(unit):
        return _COUNT
    return None


def panel_units(panel_config: dict[str, Any]) -> dict[str, str]:
    """Metric name -> unit, from a panel config."""
    tables = [
        table for source in panel_config["data source"] for table in source.values()
    ]
    return {
        name: body["unit"]
        for table in tables
        for name, body in (table.get("metric") or {}).items()
        if body.get("unit")
    }


def numeric(value: Union[int, float, str, None]) -> Optional[float]:
    """The value as a float, or None when missing or not a number."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return None if math.isnan(number) else number


def format_value(value: Union[int, float, str, None], unit: DisplayUnit) -> str:
    """A value with its unit, or 'N/A' when missing.

    Counts show as an integer below 1000 and in scientific notation above.
    """
    number = numeric(value)
    if number is None:
        return "N/A"
    if unit.kind == "count":
        return str(round(number)) if abs(number) < 1000 else f"{number:.2e}"
    return f"{number * unit.scale:.{unit.precision}f}{unit.suffix}"
