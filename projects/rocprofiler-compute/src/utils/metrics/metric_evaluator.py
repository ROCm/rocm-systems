# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""MetricEvaluator for YAML metric expression evaluation."""

from __future__ import annotations

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
from utils.metrics.noise_clamper import to_noise_clamp


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
            local_expr_context.update({"raw_pmc_df": self.raw_pmc_df})
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

            eval_result = eval(
                compile(expr, "<string>", "eval"),
                {},
                local_expr_context,
            )

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
                # Do not give warning if None is explicitly specified in expression
                if "None" not in expr:
                    console_warning(
                        f"Could not evaluate expression '{expr}' - likely "
                        "due to missing counter data."
                    )
                else:
                    console_debug(
                        f"Expression '{expr}' evaluated to None - likely "
                        "explicitly specified."
                    )
                return "N/A"
            else:
                return eval_result

        except (TypeError, NameError, KeyError) as exception:
            if "empirical_peak" in str(exception):
                console_warning(f"Missing empirical peak data: {exception}.")
                return "N/A"
            else:
                console_warning(f"Failed to evaluate expression '{expr}': {exception}.")
                return "N/A"

        except AttributeError as attribute_error:
            console_warning(
                f"Failed to evaluate expression '{expr}': {attribute_error}."
            )
            return "N/A"

        except pd.errors.IntCastingNaNError as exception:
            console_warning(f"Failed to evaluate expression '{expr}': {exception}.")
            return "N/A"

        except ValueError as value_error:
            console_warning(f"Failed to evaluate expression '{expr}': {value_error}.")
            return "N/A"
