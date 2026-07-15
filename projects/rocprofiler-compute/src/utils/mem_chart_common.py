# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Shared helpers for memory chart renderers (gfx9, gfx11).
"""

from typing import Any, Optional, Union

from utils.utils_analysis import format_bw_human_readable

COLORS = {
    "kernel": "green",
    "block": "blue",
    "tcp": "cyan",
    "lds": "magenta",
    "sqc": "yellow",
    "read": "bright_cyan",
    "write": "bright_yellow",
    "atomic": "bright_magenta",
    "util": "bright_green",
    "hit": "yellow",
    "stall": "indian_red",
    "bw": "bright_cyan",
}


def format_value(
    value: Union[int, float, str, None], unit: str = "", precision: int = 1
) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value
    if unit == "%":
        return f"{value:.{precision}f}%"
    if unit in ("GB/s", "Bytes/s"):
        return format_bw_human_readable(value, unit, precision)
    return f"{value:.{precision}f}{unit}"


def format_sci(value: Union[int, float, str, None], precision: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if abs(value) < 1000:
        return f"{int(value)}"
    return f"{value:.{precision}e}"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    formatted = format_value(value, unit)
    return f"{label} [{color}]{formatted}[/{color}]"


def bar(pct: Optional[float], w: int = 10) -> str:
    if pct is None:
        return "░" * w
    try:
        pct = float(pct)
    except (ValueError, TypeError):
        return "░" * w
    filled = int(w * min(100, max(0, pct)) / 100)
    return "█" * filled + "░" * (w - filled)


def safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum non-None numeric values. Returns None if no value is valid."""
    total = 0.0
    any_valid = False
    for v in values:
        if v is not None:
            try:
                total += float(v)
                any_valid = True
            except (ValueError, TypeError):
                pass
    return total if any_valid else None


def fmt_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_sci(value):>7}"
    else:
        value_str = ""
    return f"{label_str}{value_str}"
