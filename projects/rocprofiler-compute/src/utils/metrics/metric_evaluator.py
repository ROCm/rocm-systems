# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""MetricEvaluator for YAML metric expression evaluation."""

import warnings
from collections import ChainMap
from typing import Any, Callable, Dict, List, Optional, Set, Union

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
    tables: List[Union[pd.DataFrame, ChainMap]],
    build_in_vars: Dict[str, str],
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


def has_columns(table: Union[pd.DataFrame, ChainMap], columns: Set[str]) -> bool:
    """Return whether table has every column in columns."""
    return all(column in table for column in columns)


def collectables_per_table(
    eval_strings: List[object],
    table_index_of_row: List[int],
    tables: List[Union[pd.DataFrame, ChainMap]],
    collectables: Dict[str, str],
    used_collectables: Set[str],
) -> List[Set[str]]:
    """Return the collectables to evaluate on each table.

    eval_strings[i] reads tables[table_index_of_row[i]], as counter_table_index
    chose. Callers list the pass tables first and the merged table last.
    """
    merged_index = len(tables) - 1
    collectables_to_evaluate: List[Set[str]] = [set() for _table in tables]

    # A pass table evaluates the collectables its eval strings use
    for eval_string, table_index in zip(eval_strings, table_index_of_row):
        if table_index != merged_index:
            _, row_collectables = extract_counters_and_variables(
                eval_string, collectables, include_supported_denom=False
            )
            collectables_to_evaluate[table_index].update(row_collectables)

    # The merged table takes each collectable from the first table that holds
    # its counters, so that table evaluates it, with the ones it uses
    tables_holding: Dict[str, Set[int]] = {}
    for collectable in used_collectables:
        formula = collectables[collectable]
        columns = columns_read(build_eval_string(formula), collectables)
        tables_holding[collectable] = {
            table_index
            for table_index, table in enumerate(tables)
            if has_columns(table, columns)
        }
        if tables_holding[collectable]:
            first_holding = min(tables_holding[collectable])
            _, collectables_it_uses = extract_counters_and_variables(
                formula, collectables, include_supported_denom=False
            )
            collectables_to_evaluate[first_holding].add(collectable)
            collectables_to_evaluate[first_holding].update(collectables_it_uses)

    # Leave out a collectable the expressions do not use or the table lacks
    return [
        {
            collectable
            for collectable in collectable_names
            if collectable in used_collectables
            and table_index in tables_holding[collectable]
        }
        for table_index, collectable_names in enumerate(collectables_to_evaluate)
    ]


class MetricEvaluator:
    """Encapsulates metric evaluation logic and eliminates global variables."""

    def __init__(
        self,
        raw_pmc_df: pd.DataFrame,
        sys_vars: Dict[str, Any],
        empirical_peaks: Dict[str, Any],
        warn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.raw_pmc_df = raw_pmc_df
        self.sys_vars = sys_vars
        self.empirical_peaks = empirical_peaks
        # Reports an expression that fails to evaluate; None means console_warning
        self.warn = warn

    def _warn(self, message: str) -> None:
        (self.warn or console_warning)(message)

    def eval_expression(self, expr: str) -> Union[str, float, int]:
        """Evaluate a single expression with proper local context."""
        try:
            # Create comprehensive local context
            local_expr_context: Dict[str, Any] = {}
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
                self._warn(
                    f"Missing empirical peak data: "
                    f"{type(exception).__name__}: {exception}."
                )
                return "N/A"
            else:
                self._warn(
                    f"Failed to evaluate expression '{expr}': "
                    f"{type(exception).__name__}: {exception}."
                )
                return "N/A"

        except AttributeError as attribute_error:
            self._warn(
                f"Failed to evaluate expression '{expr}': "
                f"{type(attribute_error).__name__}: {attribute_error}."
            )
            return "N/A"

        except pd.errors.IntCastingNaNError as exception:
            self._warn(
                f"Failed to evaluate expression '{expr}': "
                f"{type(exception).__name__}: {exception}."
            )
            return "N/A"

        except ValueError as value_error:
            self._warn(
                f"Failed to evaluate expression '{expr}': "
                f"{type(value_error).__name__}: {value_error}."
            )
            return "N/A"
