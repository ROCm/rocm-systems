# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Aggregation helpers used inside YAML metric expressions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.logger import console_warning


def calc_pct_of_peak(
    value: float | str | None,
    peak: float | str | None,
) -> float | None:
    """Return 100.0 * value / peak, or None on invalid, NaN, or zero-peak input."""
    if pd.isna(value) or pd.isna(peak):
        return None
    try:
        return float(value) / float(peak) * 100.0
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def to_min(*args: Any) -> float:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].min()
    elif min(args) is None:
        return np.nan
    else:
        return min(args)


def to_max(*args: Any) -> float | np.ndarray:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        return np.maximum(args[0], args[1])
    elif max(args) is None:
        return np.nan
    else:
        return max(args)


def to_avg(
    a: pd.Series | np.ndarray | list | int | float | str | np.number | None,
) -> float | np.floating:
    if a is None:
        return np.nan
    if np.isscalar(a) and pd.isna(a):
        return np.nan
    elif isinstance(a, pd.Series):
        if a.empty:
            return np.nan
        elif np.isnan(a).all():
            return np.nan
        else:
            return a.mean()
    elif isinstance(a, (np.ndarray, list)):
        arr = np.array(a)
        if arr.size == 0:
            return np.nan
        elif np.isnan(arr).all():
            return np.nan
        else:
            return np.nanmean(arr)
    elif isinstance(a, (int, float, np.number)):
        if np.isnan(a):
            return np.nan
        else:
            return float(a)
    elif isinstance(a, str):
        if not a or a == "N/A":
            return np.nan
        return float(a)
    else:
        raise Exception(f"to_avg: unsupported type: {type(a)}")


def to_median(a: pd.Series | None) -> float:
    if a is None:
        return np.nan
    if isinstance(a, pd.Series):
        if a.empty or np.isnan(a).all():
            return np.nan
        return a.median()
    raise Exception("to_median: unsupported type.")


def to_std(a: pd.Series) -> float:
    if isinstance(a, pd.Series):
        # Define std as 0.0 if there is only one element
        if len(a) <= 1:
            return 0.0
        return a.std()
    else:
        raise Exception("to_std: unsupported type.")


def to_int(
    a: int | float | str | np.integer | pd.Series | None,
) -> int | float | pd.Series:
    if a is None:
        return np.nan
    if np.isscalar(a) and pd.isna(a):
        return np.nan
    elif isinstance(a, (int, float, np.integer)):
        return int(a)
    elif isinstance(a, pd.Series):
        # "Int64" handles null values
        return a.astype("Int64")
    elif isinstance(a, str):
        return int(a)
    else:
        raise Exception("to_int: unsupported type.")


def to_sum(
    a: pd.Series | int | float | np.number | None,
) -> float:
    if a is None:
        return np.nan
    elif isinstance(a, (int, float, np.number)):
        if np.isnan(a):
            return np.nan
        return float(a)
    elif isinstance(a, pd.Series):
        if a.empty:
            return np.nan
        elif np.isnan(a).all():
            return np.nan
        return a.sum()
    else:
        raise Exception("to_sum: unsupported type.")


def to_round(a: pd.Series | float, b: int) -> pd.Series | float:
    if isinstance(a, pd.Series):
        return a.round(b)
    else:
        return round(a, b)


def to_quantile(a: pd.Series | None, b: float) -> float:
    if a is None:
        return np.nan
    elif isinstance(a, pd.Series):
        return a.quantile(b)
    else:
        raise Exception("to_quantile: unsupported type.")


def to_mod(
    a: pd.Series | float,
    b: pd.Series | float,
) -> pd.Series | float:
    if isinstance(a, pd.Series):
        return a.mod(b)
    else:
        return a % b


def to_concat(a: Any, b: Any) -> str:  # noqa: ANN401
    return str(a) + str(b)


def _sorted_dispatch_ids(series_list: list[pd.Series]) -> list[object]:
    dispatch_ids: set[object] = set()
    for series in series_list:
        dispatch_ids.update(series.index)
    return sorted(dispatch_ids, key=lambda item: (str(type(item)), str(item)))


def _fragment_value(series: pd.Series, dispatch_id: object) -> float | None:
    if dispatch_id not in series.index:
        return None
    value = series.loc[dispatch_id]
    if pd.isna(value):
        return None
    return float(value)


def _warn_skipped_dispatches(kind: str, used: int, total: int) -> None:
    if total and used != total:
        console_warning(
            "metrics",
            f"{kind}: used {used} of {total} dispatches; "
            "skipped dispatches with a missing or NaN fragment",
        )


def merge_dispatch_weighted_avg(
    ratio_series_list: list[pd.Series],
    weight_series_list: list[pd.Series],
) -> float:
    """Pool (M0*C0 + M1*C1) / (C0 + C1) across dispatches.

    Sum the weighted products and the weights, then divide once. A mean of
    per-dispatch ratios changes the number when dispatches differ in size.
    """
    if not ratio_series_list or len(ratio_series_list) != len(weight_series_list):
        return np.nan

    dispatch_ids = _sorted_dispatch_ids(ratio_series_list + weight_series_list)
    if not dispatch_ids:
        return np.nan

    weighted_total = 0.0
    weight_total = 0.0
    used = 0
    for dispatch_id in dispatch_ids:
        numerator = 0.0
        denominator = 0.0
        skip_dispatch = False
        for ratio_series, weight_series in zip(ratio_series_list, weight_series_list):
            weight = _fragment_value(weight_series, dispatch_id)
            ratio = _fragment_value(ratio_series, dispatch_id)
            if weight is None or ratio is None:
                skip_dispatch = True
                break
            numerator += ratio * weight
            denominator += weight
        if skip_dispatch or denominator == 0.0:
            continue
        weighted_total += numerator
        weight_total += denominator
        used += 1

    _warn_skipped_dispatches("WEIGHTED_AVG", used, len(dispatch_ids))
    if used == 0 or weight_total == 0.0:
        return np.nan
    return weighted_total / weight_total


def merge_dispatch_collect_sum(ratio_series_list: list[pd.Series]) -> float:
    """Sum absolute fragment values across dispatches.

    Do not average per-dispatch rates. Shared-denominator rates belong in
    COLLECT_RATIO, which pools numerator and denominator totals.
    """
    if not ratio_series_list:
        return np.nan

    dispatch_ids = _sorted_dispatch_ids(ratio_series_list)
    if not dispatch_ids:
        return np.nan

    total = 0.0
    used = 0
    for dispatch_id in dispatch_ids:
        dispatch_total = 0.0
        skip_dispatch = False
        for ratio_series in ratio_series_list:
            value = _fragment_value(ratio_series, dispatch_id)
            if value is None:
                skip_dispatch = True
                break
            dispatch_total += value
        if skip_dispatch:
            continue
        total += dispatch_total
        used += 1

    _warn_skipped_dispatches("COLLECT_SUM", used, len(dispatch_ids))
    if used == 0:
        return np.nan
    return total


def merge_dispatch_collect_ratio(
    numerator_series_list: list[pd.Series],
    denominator_series_list: list[pd.Series],
) -> float:
    """Sum numerators and denominators across dispatches, then divide once."""
    if not numerator_series_list or not denominator_series_list:
        return np.nan

    all_series = numerator_series_list + denominator_series_list
    dispatch_ids = _sorted_dispatch_ids(all_series)
    if not dispatch_ids:
        return np.nan

    numerator_total = 0.0
    denominator_total = 0.0
    used = 0
    for dispatch_id in dispatch_ids:
        numerator = 0.0
        denominator = 0.0
        skip_dispatch = False
        for series in numerator_series_list:
            value = _fragment_value(series, dispatch_id)
            if value is None:
                skip_dispatch = True
                break
            numerator += value
        if not skip_dispatch:
            for series in denominator_series_list:
                value = _fragment_value(series, dispatch_id)
                if value is None:
                    skip_dispatch = True
                    break
                denominator += value
        if skip_dispatch or denominator == 0.0:
            continue
        numerator_total += numerator
        denominator_total += denominator
        used += 1

    _warn_skipped_dispatches("COLLECT_RATIO", used, len(dispatch_ids))
    if used == 0 or denominator_total == 0.0:
        return np.nan
    return numerator_total / denominator_total
