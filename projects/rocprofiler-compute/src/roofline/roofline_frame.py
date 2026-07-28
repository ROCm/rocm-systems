# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Translate a figure's data (kernel dots, roof knees, compute
ceilings, bandwidths) into the log-log axes it opens on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

FRAME_PAD = 1.6
FRAME_MIN_DECADES = 2.5
FRAME_SLOPE_SKEW = 2.0
FRAME_NOMINAL_ASPECT = 1.5


@dataclass
class FrameAnchors:
    """What a frame has to hold. Compute ceilings are throughputs rather than
    points because those lines run off the right edge, bounding y alone."""

    points: list[tuple[float, float]] = field(default_factory=list)
    throughputs: list[float] = field(default_factory=list)
    bandwidths: list[float] = field(default_factory=list)


def frame_bounds(
    anchors: FrameAnchors,
    aspect: float = FRAME_NOMINAL_ASPECT,
) -> Optional[tuple[float, float, float, float]]:
    """Frame these anchors as (x_lo, x_hi, y_lo, y_hi), or None when they leave
    nothing to frame. aspect is the plot area's width over its height."""
    points = [(ai, perf) for ai, perf in anchors.points if ai > 0 and perf > 0]
    xs = [ai for ai, _ in points]
    ys = [perf for _, perf in points]
    ys += [perf for perf in anchors.throughputs if perf > 0]
    if not xs or not ys:
        return None

    # Anchor where each diagonal reaches the lowest throughput, not just its
    # knee: a stacked figure's tallest ceiling lifts the knees decades above the
    # kernels, and framing knees alone clips the steeper roofs.
    perf_lo = min(ys)
    slopes = [math.log10(bw) for bw in anchors.bandwidths if bw > 0]
    xs += [perf_lo / bandwidth for bandwidth in anchors.bandwidths if bandwidth > 0]

    x_range = _padded_log_span(min(xs), max(xs))
    y_range = _padded_log_span(min(ys), max(ys))
    # x alone, since _shaped_to_aspect sizes y against whatever x ends up being.
    x_range = _widened_to(x_range, FRAME_MIN_DECADES)
    x_range, y_range = _shaped_to_aspect(x_range, y_range, aspect)
    x_range, y_range = _pinned_to_slopes(x_range, y_range, slopes, aspect)
    return (10 ** x_range[0], 10 ** x_range[1], 10 ** y_range[0], 10 ** y_range[1])


def _padded_log_span(lo: float, hi: float) -> tuple[float, float]:
    """A positive [lo, hi] in log10, with FRAME_PAD of blank on each side."""
    pad = math.log10(FRAME_PAD)
    return (math.log10(lo) - pad, math.log10(hi) + pad)


def _widened_to(span: tuple[float, float], decades: float) -> tuple[float, float]:
    """Widen a log10 span about its midpoint to cover decades. Widens only, so
    nothing already framed falls back out."""
    lo, hi = span
    if hi - lo >= decades:
        return (lo, hi)
    mid = 0.5 * (lo + hi)
    return (mid - 0.5 * decades, mid + 0.5 * decades)


def _pinned_to_slopes(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    slopes: list[float],
    aspect: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Hold the left edge out until every roof crosses the bottom edge inside the
    frame: x_lo + log10(bandwidth) <= y_lo. The top edge pays for that width,
    since lowering the floor is what pushes slopes off the left edge."""
    if not slopes:
        return x_range, y_range
    x_range = (min(x_range[0], y_range[0] - max(slopes)), x_range[1])
    if not aspect > 0:
        return x_range, y_range
    room_for_slope = (x_range[1] - x_range[0]) / (aspect * FRAME_SLOPE_SKEW)
    if room_for_slope > y_range[1] - y_range[0]:
        y_range = (y_range[0], y_range[0] + room_for_slope)
    return x_range, y_range


def _shaped_to_aspect(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    aspect: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Widen whichever axis is cramped until a roof reads within
    FRAME_SLOPE_SKEW of 45 degrees, which turns on the decades per pixel each
    axis shows and so on the window's shape."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    if not (x_span > 0 and y_span > 0 and aspect > 0):
        return x_range, y_range
    screen_slope = x_span / (aspect * y_span)
    if screen_slope > FRAME_SLOPE_SKEW:
        return x_range, _widened_to(y_range, x_span / (aspect * FRAME_SLOPE_SKEW))
    if screen_slope < 1 / FRAME_SLOPE_SKEW:
        return _widened_to(x_range, (aspect * y_span) / FRAME_SLOPE_SKEW), y_range
    return x_range, y_range
