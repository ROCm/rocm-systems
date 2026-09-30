# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Expression evaluation shared by database analysis call sites."""

import ast
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
from utils.metrics.expression import transform_expression
from utils.metrics.noise_clamper import get_noise_clamp_warnings, to_noise_clamp
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars

# Analysis evaluates expressions once per kernel, so the same expression problem is
# hit again for every kernel. Collect the messages here and report each
# distinct one once, at the end of pre_processing.
_na_expression_messages: dict[str, str] = {}
_failed_expression_messages: set[str] = set()


def record_na_expression(metric_name: str, message: str) -> None:
    """Record that an expression for metric_name evaluated to N/A."""
    _na_expression_messages.setdefault(metric_name, message)


def record_failed_expression(message: str) -> None:
    """Record that an expression could not be evaluated."""
    _failed_expression_messages.add(message)


def report_evaluation_diagnostics() -> None:
    """Report each collected evaluation message once and clear the collection.

    N/A goes to debug because the N/A itself is already in the analyze output.
    A failure stays a warning because output cannot tell a missing counter
    apart from a counter that evaluated to nothing.
    """
    for message in _na_expression_messages.values():
        console_debug(message)
    for message in sorted(_failed_expression_messages):
        console_warning(message)
    _na_expression_messages.clear()
    _failed_expression_messages.clear()


def evaluate(
    name: str,
    value: str,
    pmc_df: pd.DataFrame,
    sys_info: dict[str, Any],  # noqa ANN401
    parse: bool = False,
    emit_variance_warnings: bool = False,
) -> Any:  # noqa ANN401
    if parse:
        original_value = value
        value = re.sub(
            r"\$([0-9A-Za-z_]+)",
            lambda m: f'sys_info["{m.group(1)}"]',
            value,
        )
        ast_node = ast.parse(value)
        if not transform_expression(ast_node, original_value):
            return None
        value = ast.unparse(ast_node)
        value = value.replace("raw_pmc_df", "pmc_df")
        value = value.replace("pmc_df['sys_info']", "sys_info")
    else:
        value = value.replace("raw_pmc_df", "pmc_df")
        value = re.sub(
            "ammolite__([0-9A-Za-z_]+)",
            lambda m: f'sys_info["{m.group(1)}"]',
            value,
        )
    try:
        prev_noise_clamp_count = get_noise_clamp_warnings()["count"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            eval_result = eval(
                compile(value, "<string>", "eval"),
                {"__builtins__": EVAL_BUILTINS},
                {
                    # only locals
                    "pmc_df": pmc_df,
                    "sys_info": sys_info,
                    "to_avg": to_avg,
                    "to_concat": to_concat,
                    "to_int": to_int,
                    "to_max": to_max,
                    "to_median": to_median,
                    "to_min": to_min,
                    "to_mod": to_mod,
                    "to_quantile": to_quantile,
                    "to_round": to_round,
                    "to_std": to_std,
                    "to_sum": to_sum,
                    "to_noise_clamp": to_noise_clamp,
                },
            )
        # RuntimeWarnings (e.g. divide-by-zero) are surfaced only under --verbose
        for w in caught:
            console_debug(f"RuntimeWarning evaluating {name}: {value} - {w.message}")

        # eval_result can be None if expression has None explicitly specified
        # Do not give warning for this case and simply return None
        if eval_result is None:
            return None

        # Only return None for scalar NA values (NaN, pd.NA, +/-inf).
        # For vectors/Series, return as-is to preserve shape for downstream
        # operations. Note: pd.NA is not detected as scalar by np.isscalar()
        is_scalar_na = eval_result is pd.NA or (
            np.isscalar(eval_result) and (pd.isna(eval_result) or np.isinf(eval_result))
        )

        if is_scalar_na:
            # Skip warning when None is explicit or a RuntimeWarning
            # already explained the NA
            if "None" in value:
                console_debug(
                    f"Expression for {name}: {value} evaluated to "
                    "None - explicitly specified."
                )
            elif not caught:
                record_na_expression(
                    name,
                    f"Expression for {name}: {value} evaluated to N/A "
                    "(divide-by-zero or empty counter data).",
                )
            return None

        if (
            emit_variance_warnings
            and get_noise_clamp_warnings()["count"] > prev_noise_clamp_count
        ):
            console_warning(f"Variance corrected for metric: {name}")
        return eval_result
    except Exception as e:
        record_failed_expression(
            f"Failed to evaluate expression for {name}: {value} - "
            f"{type(e).__name__}: {e}"
        )
        return None


def calc_builtin_vars(
    pmc_df: pd.DataFrame,
    sys_info: dict,
    expressions: list[str],
) -> None:
    """Evaluate arch-specific built-in variables referenced by expressions
    (numActiveCUs, etc.). Mutates ``sys_info`` in place."""
    gpu_series = mi_gpu_specs.get_gpu_series(sys_info["gpu_arch"])
    _, expression_builtin_vars = extract_counters_and_variables(
        "\n".join(expressions), gpu_series
    )
    build_in_vars = {
        k: v
        for k, v in get_build_in_vars(gpu_series).items()
        if k in expression_builtin_vars
    }
    # Calculate PER_XCD variables first
    for key, value in build_in_vars.items():
        if "PER_XCD" in key:
            sys_info[key] = evaluate(key, value, pmc_df, sys_info, parse=True)
    # Variable dependent on PER_XCD variables
    for key, value in build_in_vars.items():
        if "PER_XCD" not in key:
            sys_info[key] = evaluate(key, value, pmc_df, sys_info, parse=True)
