# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Plotly hover-tooltip HTML for the roofline figure."""

import html
from typing import Optional

KERNEL_NAME_FONT_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_HOVER_WRAP_WIDTH = 44


def truncate_kernel_name(name: str) -> str:
    """Cap a kernel name at 200 characters so a pathological name can't blow
    up the tooltip."""
    max_length = 200
    suffix = "..."
    if len(name) <= max_length:
        return name
    return name[: max_length - len(suffix)] + suffix


def wrap_hover_name(name: str) -> str:
    """Wrap a kernel name across as many tooltip lines as it takes."""
    if not name:
        return ""
    lines = [
        html.escape(name[start : start + _HOVER_WRAP_WIDTH], quote=False)
        for start in range(0, len(name), _HOVER_WRAP_WIDTH)
    ]
    return (
        f'<span style="font-family:{KERNEL_NAME_FONT_FAMILY}">'
        + "<br>".join(lines)
        + "</span>"
    )


def build_kernel_hover_template(
    name_html: str,
    limiter: str,
    limiter_category: str,
    count: Optional[float],
    total_dispatches: Optional[float],
    total_time: Optional[float],
    time_unit: str,
    pct_runtime: Optional[float],
    ops_flops: str,
) -> str:
    """Kernel hover template; per-point values come from customdata."""
    unit = f"G{ops_flops}s/s"
    kernel_time_txt = format_hover_number(total_time, ",.0f")
    total_app_time = (
        total_time / (pct_runtime / 100.0)
        if total_time is not None and pct_runtime
        else None
    )
    total_app_time_txt = format_hover_number(total_app_time, ",.0f")
    pct_runtime_txt = format_hover_number(pct_runtime, ",.2f")
    pct_dispatches = (
        100.0 * count / total_dispatches
        if count is not None and total_dispatches
        else None
    )
    pct_dispatches_txt = format_hover_number(pct_dispatches, ",.2f")
    return _hover(
        name_html,
        [
            f"<b>Limited by {limiter_category}: {limiter}</b>",
            f"Performance: %{{customdata[1]}}% (%{{y:,.0f}} / %{{customdata[0]}} {unit})",
            "%{customdata[2]}",
            f"Dispatch Number: {pct_dispatches_txt}% ({_format_integer(count)} / {_format_integer(total_dispatches)})",
            f"Duration: {pct_runtime_txt}% ({kernel_time_txt} / {total_app_time_txt} {time_unit})",
            "AI: %{x:.6g}",
        ],
    )


def build_roof_hover(
    level_label: str,
    bandwidth: float,
    compute_peaks: list[tuple[str, float]],
    ops_flops: str,
) -> str:
    """Memory-bandwidth-roof hover: name, model, slope, and every flat compute
    roof this slope caps against, each labeled with its datatype."""
    rows = [
        "Model: throughput = min(bandwidth \u00d7 AI, compute peak).",
        f"Bandwidth (slope): {format_bandwidth(bandwidth)}",
    ]
    if compute_peaks:
        rows.append("Compute peaks (flat roofs):")
        for label, value in compute_peaks:
            rows.append(
                f"\u2003{label}: {format_hover_number(value, ',.2f')} G{ops_flops}s/s"
            )
    return _hover(f"{level_label} bandwidth roofline", rows)


def build_compute_peak_hover(
    label: str, value: float, ops_flops: str, dtype: str
) -> str:
    """Flat compute-peak-line hover, in the same shape as the memory-roof hover."""
    return _hover(
        f"{dtype} {label} compute peak",
        [
            "Model: throughput \u2264 compute peak (flat roof).",
            f"Peak throughput: {format_hover_number(value, ',.2f')} G{ops_flops}s/s",
        ],
    )


def format_hover_number(value: object, spec: str) -> str:
    """Format a numeric tooltip value with the given format spec, or N/A."""
    if value is None:
        return "N/A"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "N/A"


def _format_integer(value: object) -> str:
    """Thousands-separated integer for the tooltip, or N/A when missing."""
    if value is None:
        return "N/A"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "N/A"


def _hover(header: str, rows: list[str]) -> str:
    """Assemble the common Plotly hover body."""
    return "<br>".join([header, "", *rows]) + "<extra></extra>"


def format_bandwidth(gb_per_s: float) -> str:
    """Bandwidth as GB/s, switching to TB/s at >= 1000 GB/s
    so the roof hover stays readable."""
    try:
        value = float(gb_per_s)
    except (TypeError, ValueError):
        return "N/A"
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:,.3f} TB/s"
    return f"{value:,.3f} GB/s"
