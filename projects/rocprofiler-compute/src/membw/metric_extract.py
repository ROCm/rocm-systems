# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Extract pre-computed metric values from panel 3000 DataFrames."""

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from membw.models import MEMBW_TABLE_IDS

_METRIC_COL = "Metric"
_VALUE_COL = "Avg"
_UNIT_COL = "Unit"


@dataclass(frozen=True)
class MetricExtractionResult:
    """Combined result of a single-pass metric extraction."""

    values: dict[str, Optional[float]]
    units: dict[str, str]
    availability: str
    availability_reason: Optional[str]


def extract_membw_metrics(
    dfs: dict[int, pd.DataFrame],
    metric_keys: frozenset[str],
) -> MetricExtractionResult:
    """Extract values, units, and availability in a single pass."""
    available_table_ids = [tid for tid in MEMBW_TABLE_IDS if tid in dfs]
    if not available_table_ids:
        return MetricExtractionResult(
            values={key: None for key in metric_keys},
            units={},
            availability="unavailable",
            availability_reason="no panel 3000 data",
        )

    all_values: dict[str, Optional[float]] = {}
    all_units: dict[str, str] = {}

    for table_id in available_table_ids:
        df = dfs[table_id]
        has_metric = _METRIC_COL in df.columns
        has_value = _VALUE_COL in df.columns
        has_unit = _UNIT_COL in df.columns
        if not has_metric:
            continue
        cols = list(
            zip(
                df[_METRIC_COL],
                df[_VALUE_COL] if has_value else [None] * len(df),
                df[_UNIT_COL] if has_unit else [""] * len(df),
            )
        )
        for name, value, unit in cols:
            if name not in all_values and has_value:
                all_values[name] = _sanitize_value(value)
            if name not in all_units and has_unit:
                all_units[name] = str(unit)

    values = {key: all_values.get(key) for key in metric_keys}
    present = metric_keys & frozenset(k for k, v in all_values.items() if v is not None)
    if present == metric_keys:
        availability = "full"
        availability_reason = None
    else:
        missing = sorted(metric_keys - present)
        availability = "partial"
        availability_reason = f"missing: {', '.join(missing)}"

    return MetricExtractionResult(
        values=values,
        units=all_units,
        availability=availability,
        availability_reason=availability_reason,
    )


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
