# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Extract pre-computed metric values from panel 3000 DataFrames."""

import math
from typing import Optional

import pandas as pd

from membw.models import MEMBW_TABLE_IDS

_METRIC_COL = "Metric"
_VALUE_COL = "Avg"
_UNIT_COL = "Unit"


def extract_metric_values(
    dfs: dict[int, pd.DataFrame],
    metric_keys: frozenset[str],
) -> dict[str, Optional[float]]:
    """Look up each metric key in panel 3000 DataFrames.

    Missing keys or NaN values map to None.
    """
    lookup = _build_metric_lookup(dfs)
    return {key: lookup.get(key) for key in metric_keys}


def extract_metric_units(
    dfs: dict[int, pd.DataFrame],
) -> dict[str, str]:
    """Build a metric-key-to-unit mapping from panel 3000 DataFrames."""
    units: dict[str, str] = {}
    for table_id in MEMBW_TABLE_IDS:
        if table_id not in dfs:
            continue
        df = dfs[table_id]
        if _METRIC_COL not in df.columns or _UNIT_COL not in df.columns:
            continue
        for name, unit in zip(df[_METRIC_COL], df[_UNIT_COL]):
            if name not in units:
                units[name] = str(unit)
    return units


def check_metric_availability(
    dfs: dict[int, pd.DataFrame],
    metric_keys: frozenset[str],
) -> tuple[str, Optional[str]]:
    """Determine data availability for tree evaluation."""
    available_table_ids = [tid for tid in MEMBW_TABLE_IDS if tid in dfs]
    if not available_table_ids:
        return ("unavailable", "no panel 3000 data")

    lookup = _build_metric_lookup(dfs)
    present = metric_keys & frozenset(lookup)
    if present == metric_keys:
        return ("full", None)
    missing = sorted(metric_keys - present)
    return ("partial", f"missing: {', '.join(missing)}")


# --- Private helpers ---


def _build_metric_lookup(
    dfs: dict[int, pd.DataFrame],
) -> dict[str, Optional[float]]:
    """Build a flat metric-key-to-value mapping across MEMBW tables."""
    result: dict[str, Optional[float]] = {}
    for table_id in MEMBW_TABLE_IDS:
        if table_id not in dfs:
            continue
        df = dfs[table_id]
        if _METRIC_COL not in df.columns or _VALUE_COL not in df.columns:
            continue
        for name, value in zip(df[_METRIC_COL], df[_VALUE_COL]):
            if name not in result:
                result[name] = _sanitize_value(value)
    return result


def _sanitize_value(value: object) -> Optional[float]:
    """Convert a DataFrame cell to Optional[float]. None for NaN/inf/non-numeric."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return as_float
