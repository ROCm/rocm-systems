# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Plotly hover-tooltip HTML for the roofline figure.

All tooltip markup lives here so ``roofline_main`` builds figures at one
abstraction level and the (HTML-flavored) string assembly stays in one place.
Each ``build_*`` function returns a Plotly ``hovertemplate`` body.
"""

from typing import Any, Optional

from roofline.roofline_shared import KERNEL_NAME_FONT_FAMILY

# Kernel names wrap at this width so a long name stays readable in the tooltip.
_HOVER_WRAP_WIDTH = 44
# Bandwidth switches from GB/s to TB/s at/above this many GB/s.
_BANDWIDTH_TB_THRESHOLD_GBPS = 1000.0


def wrap_hover_name(name: str, width: int = _HOVER_WRAP_WIDTH) -> str:
    """Wrap a full kernel name so long names stay readable in the tooltip."""
    if not name:
        return ""
    chunks = [name[i : i + width] for i in range(0, len(name), width)]
    escaped = [
        chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for chunk in chunks
    ]
    return (
        f'<span style="font-family:{KERNEL_NAME_FONT_FAMILY}">'
        + "<br>".join(escaped)
        + "</span>"
    )


def build_point_hover(
    name_html: str,
    point: dict[str, Any],
    limiter: str,
    count: Optional[float],
    total_time: Optional[float],
    time_unit: str,
    pct_runtime: Optional[float],
    ops_flops: str,
) -> str:
    """Kernel-point hover: the name, then the achieved/peak
    throughput, percent of roofline, limiter, dispatches, and runtime."""
    unit = f"G{ops_flops}s/s"
    time_txt = (
        f"{_num(total_time, ',.2f')} {time_unit}".strip()
        if total_time is not None
        else "N/A"
    )
    return _hover(
        name_html,
        [
            f"AI: {_num(point.get('ai'), '.6g')}",
            f"Achieved throughput: {_num(point.get('perf'), '.3f')} {unit}",
            f"Peak throughput: {_num(point.get('peakPerf'), '.3f')} {unit}",
            f"Percent of roofline achieved: {_num(point.get('pctRoof'), '.4f')} %",
            f"Performance limiter: {limiter}",
            f"Total dispatches: {_fmt_hover_int(count)}",
            f"Aggregate time in kernel: {time_txt}",
            f"Aggregate percent runtime: {_num(pct_runtime, '.5f')} %",
        ],
    )


def build_roof_hover(
    level_label: str,
    bandwidth: float,
    compute_peaks: list[tuple[str, float]],
    ops_flops: str,
) -> str:
    """Memory-bandwidth-roof hover: name, model, slope, and every flat compute
    roof this slope caps against, each labeled with its datatype.
    The slope is drawn up to the tallest of them."""
    rows = [
        "Model: throughput = min(bandwidth \u00d7 AI, compute peak).",
        f"Bandwidth (slope): {_format_bandwidth(bandwidth)}",
    ]
    if compute_peaks:
        rows.append("Compute peaks (flat roofs):")
        for label, value in compute_peaks:
            rows.append(f"\u2003{label}: {_num(value, ',.2f')} G{ops_flops}s/s")
    return _hover(f"{level_label} bandwidth roofline", rows)


def build_compute_peak_hover(
    label: str, value: float, ops_flops: str, dtype: str = ""
) -> str:
    """Flat compute-peak-line hover, in the same shape as the memory-roof hover."""
    title = f"{dtype} {label}".strip()
    return _hover(
        f"{title} compute peak",
        [
            "Model: throughput \u2264 compute peak (flat roof).",
            f"Peak throughput: {_num(value, ',.2f')} G{ops_flops}s/s",
        ],
    )


def _num(value: object, spec: str) -> str:
    """Format a numeric tooltip value with the given format spec, or N/A."""
    if value is None:
        return "N/A"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "N/A"


def _fmt_hover_int(value: object) -> str:
    """Thousands-separated integer for the tooltip, or N/A when missing."""
    if value is None:
        return "N/A"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "N/A"


def _hover(header: str, rows: list[str]) -> str:
    """Assemble the common Plotly hover body without bold styling."""
    return "<br>".join([header, "", *rows]) + "<extra></extra>"


def _format_bandwidth(gb_per_s: float) -> str:
    """Bandwidth as GB/s, switching to TB/s at >= 1000 GB/s
    so the roof hover stays readable."""
    try:
        value = float(gb_per_s)
    except (TypeError, ValueError):
        return "N/A"
    if abs(value) >= _BANDWIDTH_TB_THRESHOLD_GBPS:
        return f"{value / 1000.0:,.3f} TB/s"
    return f"{value:,.3f} GB/s"
