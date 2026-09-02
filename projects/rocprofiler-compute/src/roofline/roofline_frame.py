# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Derive run-invariant roofline axes from machine ceilings."""

import math
from typing import Iterable, List, Optional, Tuple

FRAME_X_MIN = 1e-2


def canonical_frame(
    bandwidths: List[float], peaks: List[float]
) -> Optional[Tuple[float, float, float, float]]:
    """Return decade-aligned bounds derived only from machine ceilings."""
    positive_bandwidths = [
        float(value) for value in bandwidths if math.isfinite(value) and value > 0
    ]
    positive_peaks = [
        float(value) for value in peaks if math.isfinite(value) and value > 0
    ]
    if not positive_bandwidths or not positive_peaks:
        return None

    minimum_bandwidth = min(positive_bandwidths)
    maximum_peak = max(positive_peaks)
    x_high = 10 ** math.ceil(math.log10(maximum_peak / minimum_bandwidth))
    y_low = 10 ** math.floor(math.log10(FRAME_X_MIN * minimum_bandwidth))
    y_high = 10 ** math.ceil(math.log10(maximum_peak))

    if x_high <= FRAME_X_MIN:
        x_high = FRAME_X_MIN * 10
    if y_high <= y_low:
        y_high = y_low * 10

    return FRAME_X_MIN, x_high, y_low, y_high


def points_outside_frame(
    frame: Tuple[float, float, float, float],
    points: Iterable[Tuple[float, float]],
) -> List[Tuple[int, float, float]]:
    """The (ai, perf) points the frame does not reach, as (index, x, y).

    The x and y values say how far past an edge that axis sits, in decades:
    negative below the low bound, positive above the high bound, and 0.0 when
    that axis does hold the point. Points are only reported, never moved to
    fit: the frame belongs to the machine, not to the run.
    """
    x_low, x_high, y_low, y_high = frame
    outside: List[Tuple[int, float, float]] = []
    for index, (ai, perf) in enumerate(points):
        x_overflow = _axis_overflow(ai, x_low, x_high)
        y_overflow = _axis_overflow(perf, y_low, y_high)
        if x_overflow or y_overflow:
            outside.append((index, x_overflow, y_overflow))
    return outside


def _axis_overflow(value: float, low: float, high: float) -> float:
    """Signed decades a value sits past one axis's bounds; 0.0 when inside.

    A value a log axis cannot place at all overflows without limit.
    """
    if not math.isfinite(value) or value <= 0:
        return -math.inf
    if value < low:
        return math.log10(value / low)
    if value > high:
        return math.log10(value / high)
    return 0.0
