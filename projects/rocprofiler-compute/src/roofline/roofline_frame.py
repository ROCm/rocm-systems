# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Translate machine ceilings into the log-log axes a roofline opens on."""

import math
from typing import Optional

FRAME_X_MIN = 1e-2


def canonical_frame(
    bandwidths: list[float], peaks: list[float]
) -> Optional[tuple[float, float, float, float]]:
    """Return decade-aligned bounds derived only from machine ceilings."""
    valid_bandwidths = [bw for bw in bandwidths if math.isfinite(bw) and bw > 0]
    valid_peaks = [peak for peak in peaks if math.isfinite(peak) and peak > 0]
    if not valid_bandwidths or not valid_peaks:
        return None

    minimum_bandwidth = min(valid_bandwidths)
    maximum_peak = max(valid_peaks)

    log_minimum_bandwidth = math.log10(minimum_bandwidth)
    log_maximum_peak = math.log10(maximum_peak)
    log_frame_x_min = math.log10(FRAME_X_MIN)
    log_y_low_target = log_frame_x_min + log_minimum_bandwidth

    x_high_exponent = int(math.ceil(log_maximum_peak - log_minimum_bandwidth))
    y_low_exponent = int(math.floor(log_y_low_target))
    y_high_exponent = int(math.ceil(log_maximum_peak))

    min_x_high_exponent = int(log_frame_x_min + 1.0)
    if x_high_exponent <= int(math.floor(log_frame_x_min)):
        x_high_exponent = min_x_high_exponent
    if y_high_exponent <= y_low_exponent:
        y_high_exponent = y_low_exponent + 1

    while not _x_high_covers_peak(
        x_high_exponent, minimum_bandwidth, log_minimum_bandwidth, maximum_peak
    ):
        x_high_exponent += 1
        if _decade_bound(float(x_high_exponent)) is None:
            return None

    while not _y_high_covers_peak(y_high_exponent, maximum_peak):
        y_high_exponent += 1
        if _decade_bound(float(y_high_exponent)) is None:
            return None

    while not _y_low_within_target(y_low_exponent, minimum_bandwidth, log_y_low_target):
        y_low_exponent -= 1
        if _decade_bound(float(y_low_exponent)) is None:
            return None

    x_high = _decade_bound(float(x_high_exponent))
    y_low = _decade_bound(float(y_low_exponent))
    y_high = _decade_bound(float(y_high_exponent))
    if x_high is None or y_low is None or y_high is None:
        return None

    return (FRAME_X_MIN, x_high, y_low, y_high)


def _decade_bound(exponent: float) -> Optional[float]:
    """Return 10**exponent when it is a positive finite float."""
    if not math.isfinite(exponent):
        return None
    try:
        value = math.pow(10.0, exponent)
    except OverflowError:
        return None
    if value <= 0.0 or not math.isfinite(value):
        return None
    return value


def _x_high_covers_peak(
    x_high_exponent: int,
    minimum_bandwidth: float,
    log_minimum_bandwidth: float,
    maximum_peak: float,
) -> bool:
    x_high = _decade_bound(float(x_high_exponent))
    if x_high is None:
        return False
    product = x_high * minimum_bandwidth
    if math.isfinite(product) and product > 0.0:
        return product >= maximum_peak
    return (float(x_high_exponent) + log_minimum_bandwidth) >= math.log10(maximum_peak)


def _y_high_covers_peak(y_high_exponent: int, maximum_peak: float) -> bool:
    y_high = _decade_bound(float(y_high_exponent))
    if y_high is None:
        return False
    return y_high >= maximum_peak


def _y_low_within_target(
    y_low_exponent: int,
    minimum_bandwidth: float,
    log_y_low_target: float,
) -> bool:
    y_low = _decade_bound(float(y_low_exponent))
    if y_low is None:
        return False
    target = FRAME_X_MIN * minimum_bandwidth
    if math.isfinite(target) and target > 0.0:
        return y_low <= target
    return float(y_low_exponent) <= math.floor(log_y_low_target)
