# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""MetricEvaluator for YAML metric expression evaluation."""

from __future__ import annotations

import warnings
from collections import ChainMap
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
from utils.metrics.expression import build_eval_string, columns_read
from utils.utils_counter_defs import extract_counters_and_variables


def counter_table_index(
    eval_string: object,
    tables: list[pd.DataFrame | ChainMap],
    build_in_vars: dict[str, str],
) -> int:
    """Return the index of the table an eval string should read.

    Callers list the pass tables first and the merged table last. The first
    pass table that has every column the eval string reads wins. Otherwise
    the merged table wins.
    """
    if isinstance(eval_string, str):
        columns = columns_read(eval_string, build_in_vars)
        for index, table in enumerate(tables[:-1]):
            if has_columns(table, columns):
                return index
    return len(tables) - 1


def has_columns(table: pd.DataFrame | ChainMap, columns: set[str]) -> bool:
    """Return whether table has every column in columns."""
    return all(column in table for column in columns)


def collectables_per_table(
    eval_strings: list[object],
    table_indexes: list[int],
    tables: list[pd.DataFrame | ChainMap],
    collectables: dict[str, str],
    used: set[str],
) -> list[set[str]]:
    """Return the collectables to evaluate on each table.

    eval_strings[i] reads tables[table_indexes[i]], as counter_table_index
    chose. Callers list the pass tables first and the merged table last.
    """
    merged_index = len(tables) - 1
    to_evaluate: list[set[str]] = [set() for _table in tables]

    # A pass table evaluates the collectables its eval strings use
    for eval_string, index in zip(eval_strings, table_indexes):
        if index != merged_index:
            to_evaluate[index].update(_collectables_used(eval_string, collectables))

    # The merged table takes each collectable from the first table that holds
    # its counters, so that table evaluates it, with the ones it uses
    holders: dict[str, set[int]] = {}
    for name in used:
        columns = columns_read(build_eval_string(collectables[name]), collectables)
        holders[name] = {
            index for index, table in enumerate(tables) if has_columns(table, columns)
        }
        if holders[name]:
            first_holder = min(holders[name])
            to_evaluate[first_holder].add(name)
            to_evaluate[first_holder].update(
                _collectables_used(collectables[name], collectables)
            )

    # Leave out a collectable the expressions do not use or the table lacks
    return [
        {name for name in names if name in used and index in holders[name]}
        for index, names in enumerate(to_evaluate)
    ]


def _collectables_used(text: object, collectables: dict[str, str]) -> set[str]:
    """Return the collectables text uses, and the ones those use in turn."""
    _, names = extract_counters_and_variables(
        str(text), collectables, include_supported_denom=False
    )
    return names


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
