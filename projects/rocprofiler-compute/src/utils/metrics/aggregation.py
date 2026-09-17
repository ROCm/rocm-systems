# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Aggregation helpers used inside YAML metric expressions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def merge_dispatch_weighted_avg(
    ratio_series_list: list[pd.Series],
    weight_series_list: list[pd.Series],
) -> float:
    """Combine per-dispatch submetric ratios with weight counters.

    For each dispatch index present in all series, compute
    M_i = sum_k(M_{k,i} * C_{k,i}) / sum_k(C_{k,i}), then aggregate
    dispatch values with to_avg (avg-only Phase 2 semantics).
    """
    if not ratio_series_list or len(ratio_series_list) != len(weight_series_list):
        return np.nan

    all_series = ratio_series_list + weight_series_list
    index_sets = [set(series.index) for series in all_series]
    common_idx = set.intersection(*index_sets) if index_sets else set()
    if not common_idx:
        return np.nan

    dispatch_values: list[float] = []
    for dispatch_id in sorted(common_idx, key=lambda x: (str(type(x)), x)):
        numerator = 0.0
        denominator = 0.0
        skip_dispatch = False
        for ratio_series, weight_series in zip(
            ratio_series_list, weight_series_list, strict=True
        ):
            weight = weight_series.loc[dispatch_id]
            ratio = ratio_series.loc[dispatch_id]
            if pd.isna(weight) or pd.isna(ratio):
                skip_dispatch = True
                break
            numerator += float(ratio) * float(weight)
            denominator += float(weight)
        if skip_dispatch or denominator == 0.0:
            continue
        dispatch_values.append(numerator / denominator)

    if not dispatch_values:
        return np.nan
    return float(to_avg(pd.Series(dispatch_values)))


def merge_dispatch_collect_sum(ratio_series_list: list[pd.Series]) -> float:
    """Per dispatch sum submetric ratios, then run-level avg (M = h + i)."""
    if not ratio_series_list:
        return np.nan

    index_sets = [set(series.index) for series in ratio_series_list]
    common_idx = set.intersection(*index_sets) if index_sets else set()
    if not common_idx:
        return np.nan

    dispatch_values: list[float] = []
    for dispatch_id in sorted(common_idx, key=lambda x: (str(type(x)), x)):
        total = 0.0
        skip_dispatch = False
        for ratio_series in ratio_series_list:
            ratio = ratio_series.loc[dispatch_id]
            if pd.isna(ratio):
                skip_dispatch = True
                break
            total += float(ratio)
        if skip_dispatch:
            continue
        dispatch_values.append(total)

    if not dispatch_values:
        return np.nan
    return float(to_avg(pd.Series(dispatch_values)))
