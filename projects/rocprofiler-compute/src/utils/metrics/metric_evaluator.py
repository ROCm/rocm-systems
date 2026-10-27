# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""MetricEvaluator for YAML metric expression evaluation."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import console_debug, console_warning
from utils.metrics.aggregation import (
    to_avg,
    to_concat,
    to_int,
    to_max,
    to_median,
    to_min,
    to_mod,
    to_quantile,
    to_round,
    to_std,
    to_sum,
)
from utils.metrics.common import EVAL_BUILTINS
from utils.metrics.noise_clamper import to_noise_clamp

# Columns an expression reads, whether the frame is named raw_pmc_df or pmc_df.
_FRAME_COLUMN_RE = re.compile(r"(?:raw_pmc_df|pmc_df)\[['\"]([^'\"]+)['\"]\]")
_TO_SUM_RE = re.compile(r"\bto_sum\s*\(")


def rows_for_expression(frame: pd.DataFrame, expression: str) -> pd.DataFrame:
    """Keep dispatches that have every counter a summed expression reads.

    Numerator and denominator sums then use one set of rows.
    """
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or not expression
        or _TO_SUM_RE.search(expression) is None
    ):
        return frame
    columns: list[str] = []
    seen: set[str] = set()
    for column in _FRAME_COLUMN_RE.findall(expression):
        if column in seen or column not in frame.columns:
            continue
        seen.add(column)
        columns.append(column)
    if not columns:
        return frame
    complete = frame[columns].notna().all(axis=1)
    if bool(complete.all()):
        return frame
    return frame.loc[complete]


class MetricEvaluator:
    """Encapsulates metric evaluation logic and eliminates global variables."""

    def __init__(
        self,
        raw_pmc_df: pd.DataFrame,
        sys_vars: dict[str, Any],
        empirical_peaks: dict[str, Any],
    ) -> None:
        self.raw_pmc_df = raw_pmc_df
        self.sys_vars = sys_vars
        self.empirical_peaks = empirical_peaks

    def eval_expression(self, expr: str) -> str | float | int:
        """Evaluate a single expression with proper local context."""
        try:
            # Create comprehensive local context
            local_expr_context: dict[str, Any] = {}
            local_expr_context.update({
                "raw_pmc_df": rows_for_expression(self.raw_pmc_df, expr),
            })
            local_expr_context.update(self.sys_vars)
            local_expr_context.update(self.empirical_peaks)

            # Add utility functions to local context
            local_expr_context.update({
                "to_min": to_min,
                "to_max": to_max,
                "to_avg": to_avg,
                "to_median": to_median,
                "to_std": to_std,
                "to_int": to_int,
                "to_sum": to_sum,
                "to_round": to_round,
                "to_quantile": to_quantile,
                "to_mod": to_mod,
                "to_concat": to_concat,
                "to_noise_clamp": to_noise_clamp,
            })

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                eval_result = eval(
                    compile(expr, "<string>", "eval"),
                    {"__builtins__": EVAL_BUILTINS},
                    local_expr_context,
                )
            # RuntimeWarnings (e.g. divide-by-zero) are surfaced only under --verbose
            for w in caught:
                console_debug(f"RuntimeWarning evaluating '{expr}': {w.message}")

            # Only return "N/A" for scalar NA values
            # For vectors/Series, return as-is to preserve shape for
            # downstream operations
            # Note: None and pd.NA are not detected as scalar by np.isscalar()
            if (
                eval_result is None
                or eval_result is pd.NA
                or (
                    np.isscalar(eval_result)
                    and (pd.isna(eval_result) or np.isinf(eval_result))
                )
            ):
                # Skip warning when None is explicit or a RuntimeWarning
                # already explained the NA
                if "None" in expr:
                    console_debug(
                        f"Expression '{expr}' evaluated to None - explicitly specified."
                    )
                elif not caught:
                    console_debug(
                        f"Expression '{expr}' evaluated to N/A "
                        "(divide-by-zero or empty counter data)."
                    )
                return "N/A"
            else:
                return eval_result

        except (TypeError, NameError, KeyError) as exception:
            if "empirical_peak" in str(exception):
                console_warning(
                    f"Missing empirical peak data: "
                    f"{type(exception).__name__}: {exception}."
                )
                return "N/A"
            else:
                console_warning(
                    f"Failed to evaluate expression '{expr}': "
                    f"{type(exception).__name__}: {exception}."
                )
                return "N/A"

        except AttributeError as attribute_error:
            console_warning(
                f"Failed to evaluate expression '{expr}': "
                f"{type(attribute_error).__name__}: {attribute_error}."
            )
            return "N/A"

        except pd.errors.IntCastingNaNError as exception:
            console_warning(
                f"Failed to evaluate expression '{expr}': "
                f"{type(exception).__name__}: {exception}."
            )
            return "N/A"

        except ValueError as value_error:
            console_warning(
                f"Failed to evaluate expression '{expr}': "
                f"{type(value_error).__name__}: {value_error}."
            )
            return "N/A"
