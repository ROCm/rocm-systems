# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Base units for the metric values the analysis database stores.

The database stores durations in ns, bandwidth in Bytes/s, compute rates in
FLOP/s and IOP/s, and arithmetic intensity in FLOPs/Byte, all unrounded. Some
metric configs label their values with scaled units instead: GB/s, Gbps,
GFLOP/s, GFLOPs, GIOP/s, GIOPs and KB per Wave. The number a config computes
is already in its label unit, so the database multiplies it by 1e9 (or 1024
for KB per Wave) to get the base unit. The configs and the terminal display
keep their own units.

Cases to know about:
  - Gbps values are bytes per ns, the same as GB/s, despite the label.
  - GFLOPs values are rates per ns, the same as GFLOP/s, despite the label.
  - KB per Wave values are KB, and the database stores them as Bytes per Wave.
  - Percent of Peak is a ratio of two values in the same unit and keeps its scale.
  - compute_kernel_roofline_data keeps GFLOP/s and FLOPs/Byte. The HLD says the
    table keeps its shape and units until D8b replaces it, so it is not
    converted here.
"""

import numbers
from typing import Dict, Optional, Tuple, Union

# Config unit -> (base unit stored in the database, multiplier for the value)
SCALED_UNITS: Dict[str, Tuple[str, float]] = {
    "GB/s": ("Bytes/s", 1e9),
    "Gbps": ("Bytes/s", 1e9),
    "GFLOP/s": ("FLOP/s", 1e9),
    "GFLOPs": ("FLOP/s", 1e9),
    "GIOP/s": ("IOP/s", 1e9),
    "GIOPs": ("IOP/s", 1e9),
    # Scratch Allocation divides by 1024, so its value is KB and is stored as bytes.
    # 1024 is a power of two, so the scaling is exact in floating point.
    "KB per Wave": ("Bytes per Wave", 1024.0),
}

# Value names that are ratios and keep the scale of their metric.
UNSCALED_VALUE_NAMES = frozenset({"Percent of Peak"})


def base_unit(unit: Optional[str]) -> Optional[str]:
    """Return the unit a metric is stored in, given its config unit.

    Units without a scaled form, such as Bytes/s or Nanoseconds, are returned
    unchanged.
    """
    if unit in SCALED_UNITS:
        return SCALED_UNITS[unit][0]
    return unit


def to_base_unit_value(
    unit: Optional[str],
    value_name: str,
    value: Union[float, str, None],
) -> Union[float, str, None]:
    """Return one stored value in the base unit of its metric's config unit.

    Missing values, text and Percent of Peak are returned unchanged.
    """
    if unit not in SCALED_UNITS or value_name in UNSCALED_VALUE_NAMES:
        return value
    if not isinstance(value, numbers.Real):
        return value
    return value * SCALED_UNITS[unit][1]
