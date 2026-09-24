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
    kernel_dispatches_count: Optional[float],
    total_dispatches_count: Optional[float],
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
        100.0 * kernel_dispatches_count / total_dispatches_count
        if kernel_dispatches_count is not None and total_dispatches_count
        else None
    )
    pct_dispatches_txt = format_hover_number(pct_dispatches, ",.2f")
    count_txt = _format_integer(kernel_dispatches_count)
    total_dispatches_txt = _format_integer(total_dispatches_count)
    return _hover(
        name_html,
        [
            f"<b>Limited by {limiter_category}: {limiter}</b>",
            f"Performance: %{{customdata[1]}}% "
            f"(%{{y:,.0f}} / %{{customdata[0]}} {unit})",
            "AI: %{x:.6g}",
            "",
            "<b>Details</b>",
            "%{customdata[2]}",
            f"Dispatch Count: {pct_dispatches_txt}% "
            f"({count_txt} / {total_dispatches_txt})",
            f"Duration: {pct_runtime_txt}% "
            f"({kernel_time_txt} / {total_app_time_txt} {time_unit})",
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


def _safe_float(value: object) -> Optional[float]:
    """Convert to float, or None if missing/not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_hover_number(value: object, spec: str) -> str:
    """Format a numeric tooltip value with the given format spec, or N/A."""
    parsed = _safe_float(value)
    return format(parsed, spec) if parsed is not None else "N/A"


def _format_integer(value: object) -> str:
    """Thousands-separated integer for the tooltip, or N/A when missing."""
    parsed = _safe_float(value)
    return f"{int(round(parsed)):,}" if parsed is not None else "N/A"


def _hover(header: str, rows: list[str]) -> str:
    """Assemble the common Plotly hover body."""
    return "<br>".join([header, "", *rows]) + "<extra></extra>"


def _bandwidth_unit(value: float) -> tuple[float, str]:
    """Scale factor and suffix for a bandwidth magnitude: GB/s, switching to
    TB/s at >= 1000 GB/s so the hover stays readable."""
    if abs(value) >= 1000.0:
        return 1000.0, "TB/s"
    return 1.0, "GB/s"


def format_bandwidth(gb_per_s: object) -> str:
    """Bandwidth as GB/s, switching to TB/s at >= 1000 GB/s."""
    value = _safe_float(gb_per_s)
    if value is None:
        return "N/A"
    scale, unit = _bandwidth_unit(value)
    return f"{value / scale:,.3f} {unit}"


def format_bandwidth_pair(achieved_gb_per_s: object, peak_gb_per_s: object) -> str:
    """Achieved/peak bandwidth sharing a single unit suffix (GB/s or TB/s),
    picked from the peak's magnitude, matching the single-suffix style of
    the Performance field instead of scaling each side independently."""
    achieved = _safe_float(achieved_gb_per_s)
    peak = _safe_float(peak_gb_per_s)
    reference = peak if peak else achieved
    scale, unit = _bandwidth_unit(reference) if reference is not None else (1.0, "GB/s")
    achieved_txt = format_hover_number(
        achieved / scale if achieved is not None else None, ",.3f"
    )
    peak_txt = format_hover_number(peak / scale if peak is not None else None, ",.3f")
    return f"{achieved_txt} / {peak_txt} {unit}"
