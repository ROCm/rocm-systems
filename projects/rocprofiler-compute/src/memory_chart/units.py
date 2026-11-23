# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""How the memory chart displays a metric value, by the metric's unit.

Units come from the `unit` field of each metric in the memory chart panel config
(0300_memory_chart.yaml), never from a layout. A unit test fails if a metric shown
on a chart has a unit with no rule here.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from utils.utils_common import metric_ids_by_name

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

_BYTES = DisplayUnit("value", " B", 0)

_DISPLAY_UNITS: dict[str, DisplayUnit] = {
    "Percent": PERCENT,
    "Bytes/s": DisplayUnit("bandwidth", " GB/s", 3, 1e-9),
    "Bytes per Work-item": _BYTES,
    "Bytes per Workgroup": _BYTES,
    "Cycles": DisplayUnit("value", " cycles"),
    "Registers": DisplayUnit("value", precision=0),
    "Workgroups per CU": DisplayUnit("value"),
    "Workgroups per WGP": DisplayUnit("value"),
}

# Per-normalization counts, e.g. "(Requests + $normUnit)", shown with what they count
_COUNT_UNIT = re.compile(r"^\((\w+) \+ \$normUnit\)$")
_COUNT_SHORT = {"Instructions": "instr", "Requests": "req"}


def display_unit(unit: Optional[str]) -> Optional[DisplayUnit]:
    """The display rule for *unit*, or None if it has none."""
    if unit in _DISPLAY_UNITS:
        return _DISPLAY_UNITS[unit]
    match = _COUNT_UNIT.match(unit) if unit else None
    if match is None:
        return None
    counted = match.group(1)
    return DisplayUnit("count", " " + _COUNT_SHORT.get(counted, counted.lower()))


def panel_units(panel_config: dict[str, Any]) -> dict[str, str]:
    """Metric id -> unit, from a panel config."""
    tables = [
        table for source in panel_config["data source"] for table in source.values()
    ]
    return {
        metric_id: table["metric"][name]["unit"]
        for table in tables
        for name, metric_id in metric_ids_by_name(table).items()
        if table["metric"][name].get("unit")
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
        shown = str(round(number)) if abs(number) < 1000 else f"{number:.2e}"
        return shown + unit.suffix
    return f"{number * unit.scale:.{unit.precision}f}{unit.suffix}"
