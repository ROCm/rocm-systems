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


def to_min(*args: Any) -> float | pd.Series:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].min()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        left, right = args[0], args[1]
        index = left.index if isinstance(left, pd.Series) else right.index
        if not isinstance(left, pd.Series):
            left = pd.Series(left, index=index)
        if not isinstance(right, pd.Series):
            right = pd.Series(right, index=index)
        return pd.Series(np.minimum(left, right), index=index)
    elif min(args) is None:
        return np.nan
    else:
        return min(args)


def to_max(*args: Any) -> float | pd.Series:
    if len(args) == 1 and isinstance(args[0], pd.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.Series) or isinstance(args[1], pd.Series)
    ):
        left, right = args[0], args[1]
        index = left.index if isinstance(left, pd.Series) else right.index
        if not isinstance(left, pd.Series):
            left = pd.Series(left, index=index)
        if not isinstance(right, pd.Series):
            right = pd.Series(right, index=index)
        return pd.Series(np.maximum(left, right), index=index)
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


def to_bound_ratio(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
    scale: float = 100.0,
    cap: float = 100.0,
) -> pd.Series | float:
    """Return per-row ratio * scale capped at cap (for Percent traffic/util metrics)."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num = (
            numerator
            if isinstance(numerator, pd.Series)
            else pd.Series(numerator, index=denominator.index)
        )
        den = (
            denominator
            if isinstance(denominator, pd.Series)
            else pd.Series(denominator, index=num.index)
        )
        safe_den = den.replace(0, np.nan)
        return (num / safe_den * scale).clip(upper=cap)
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return min(float(numerator) / float(denominator) * scale, cap)
