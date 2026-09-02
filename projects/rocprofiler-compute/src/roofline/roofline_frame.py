# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Derive run-invariant roofline axes from machine ceilings."""

import math
from typing import List, Optional, Tuple

FRAME_X_MIN = 1e-2

# Kept until the browser framing recipe is removed in Phase 2.
FRAME_PAD = 1.6
FRAME_MIN_DECADES = 2.5
FRAME_SLOPE_SKEW = 2.0


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
