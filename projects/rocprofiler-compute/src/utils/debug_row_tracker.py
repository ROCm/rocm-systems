##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################

"""Debug utilities for tracking and evaluating metric row expressions.

This module provides debugging tools for analyzing how metric expressions
are evaluated against raw PMC data frames.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from utils.logger import console_warning

if TYPE_CHECKING:
    from utils.parser import MetricEvaluator


_MAX_DEBUG_ROWS = 5


class DebugRowTracker:
    """Track which (df_id, row_id) combinations have been processed.

    Used to avoid printing duplicate input data when multiple expressions
    in the same row use the same input variables.
    """

    def __init__(self) -> None:
        self._seen_rows: set[tuple[object, object]] = set()

    def should_show_inputs(self, df_id: object, row_id: object) -> bool:
        """Return True if this is the first expression for this (df_id, row_id).

        Args:
            df_id: The dataframe identifier.
            row_id: The row identifier within the dataframe.

        Returns:
            True for the first expression in a row, False for subsequent ones.
        """
        key = (df_id, row_id)
        if key in self._seen_rows:
            return False
        self._seen_rows.add(key)
        return True


def _print_debug_global_vars(row_expr: str, metric_evaluator: MetricEvaluator) -> None:
    """Print global $xxx variables used in the expression."""
    matched_vars = re.findall(r"ammolite__\w+", row_expr)
    seen_vars: set[str] = set()
    for var_key in matched_vars:
        if var_key in seen_vars:
            continue
        seen_vars.add(var_key)
        # Display as $varname (strip ammolite__ prefix) to match config globals
        dollar_name = f"${var_key.replace('ammolite__', '', 1)}"
        if var_key in metric_evaluator.sys_vars:
            print(f"  {dollar_name}: {metric_evaluator.sys_vars[var_key]}")
        elif var_key in metric_evaluator.empirical_peaks:
            print(f"  {dollar_name}: {metric_evaluator.empirical_peaks[var_key]}")
        else:
            print(f"  {dollar_name}: [not found]")


def _extract_column_data(
    table_key: str, col_name: str, raw_pmc_df: pd.DataFrame | dict
) -> Optional[list[Any]]:
    """Extract column data from raw_pmc_df (dict or DataFrame)."""
    if isinstance(raw_pmc_df, dict) and table_key in raw_pmc_df:
        series = raw_pmc_df[table_key][col_name]
        return series.tolist() if hasattr(series, "tolist") else list(series)
    elif isinstance(raw_pmc_df, pd.DataFrame):
        columns = raw_pmc_df.columns
        # Handle MultiIndex columns by matching on the top-level table key
        if isinstance(columns, pd.MultiIndex):
            if table_key in columns.get_level_values(0):
                series = raw_pmc_df[table_key][col_name]
                return series.tolist() if hasattr(series, "tolist") else list(series)
        # Fallback for flat (single-level) columns
        if col_name in columns:
            series = raw_pmc_df[col_name]
            return series.tolist() if hasattr(series, "tolist") else list(series)
    return None


def _collect_debug_column_data(
    row_expr: str, raw_pmc_df: pd.DataFrame | dict
) -> tuple[list[tuple[str, Optional[list[Any]]]], int]:
    """Collect column data and compute alignment width for debug output."""
    matched_cols = re.findall(
        r"raw_pmc_df\[[\"'](\w+)[\"']\]\[[\"'](\w+)[\"']\]", row_expr
    )
    seen: set[tuple[str, str]] = set()
    rows_to_print: list[tuple[str, Optional[list[Any]]]] = []
    global_width = 0

    for table_key, col_name in matched_cols:
        if (table_key, col_name) in seen:
            continue
        seen.add((table_key, col_name))
        try:
            column_data = _extract_column_data(table_key, col_name, raw_pmc_df)
            label = f"raw_pmc_df['{table_key}']['{col_name}']"
            rows_to_print.append((label, column_data))
            if column_data is not None:
                display = column_data[:_MAX_DEBUG_ROWS]
                global_width = max(
                    global_width,
                    max((len(str(v)) for v in display), default=0),
                )
        except (KeyError, TypeError) as e:
            console_warning(
                f"Skipping entry for '{table_key}'['{col_name}']. Encountered: {e}"
            )

    return rows_to_print, global_width


def _print_debug_column_data(
    rows_to_print: list[tuple[str, Optional[list[Any]]]], global_width: int
) -> None:
    """Print collected column data with aligned formatting."""
    for label, column_data in rows_to_print:
        if column_data is not None:
            n = len(column_data)
            display_data = column_data[:_MAX_DEBUG_ROWS]
            formatted = ", ".join(str(v).rjust(global_width) for v in display_data)
            if n > _MAX_DEBUG_ROWS:
                formatted += ", ..."
            print(f"  {label}: [{formatted}]")
        else:
            print(f"  {label}: [unknown type]")


def _print_debug_inputs(
    row_expr: str,
    metric_evaluator: MetricEvaluator,
    raw_pmc_df: pd.DataFrame | dict,
    show_inputs: bool,
) -> None:
    """Print input variables and column data for debug output."""
    print("Inputs:")
    if show_inputs:
        _print_debug_global_vars(row_expr, metric_evaluator)
        rows_to_print, global_width = _collect_debug_column_data(row_expr, raw_pmc_df)
        _print_debug_column_data(rows_to_print, global_width)
    else:
        print("  The same as above.")


def _print_debug_output(row_expr: str, metric_evaluator: MetricEvaluator) -> None:
    """Evaluate and print the expression result."""
    print("\nOutput:", end=" ")
    try:
        eval_result = metric_evaluator.eval_expression(row_expr)
        print(eval_result)
    except Exception as e:
        console_warning(f"Debug evaluation failed: {e}")
    print("~" * 40)


def debug_row_tracker(
    expr: str,
    row_expr: str,
    metric_evaluator: MetricEvaluator,
    raw_pmc_df: pd.DataFrame | dict,
    *,
    show_inputs: bool = True,
) -> None:
    """Debug helper for tracking and evaluating metric row expressions.

    Args:
        expr: The original metric expression (for display purposes).
        row_expr: The fully substituted expression to evaluate.
        metric_evaluator: The MetricEvaluator instance for expression evaluation.
        raw_pmc_df: Raw PMC data (DataFrame or dict).
        show_inputs: Whether to show input variable values (default: True).
    """
    print("~" * 40 + "\nExpression:")
    print(f"{expr} = {row_expr}")
    _print_debug_inputs(row_expr, metric_evaluator, raw_pmc_df, show_inputs)
    _print_debug_output(row_expr, metric_evaluator)
