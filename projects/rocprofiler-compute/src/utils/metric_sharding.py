# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Row-wise sharding of raw PMC data for parallel metric evaluation."""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

import numpy as np
import pandas as pd

SHARD_ROW_COUNT: int = 2048


def build_shard_views(
    raw_pmc_df: Union[pd.DataFrame, dict],
    shard_row_count: int = SHARD_ROW_COUNT,
) -> list[Union[pd.DataFrame, dict]]:
    """Split raw PMC data into row chunks for parallel evaluation.

    Returns a list of views, each containing at most
    shard_row_count rows of pmc_perf data. Returns the original
    data in a single-element list when sharding is unnecessary.
    """
    pmc_perf = _extract_pmc_perf(raw_pmc_df)

    if pmc_perf is None or len(pmc_perf) <= shard_row_count:
        return [raw_pmc_df]

    row_count = len(pmc_perf)
    shard_views: list[Union[pd.DataFrame, dict]] = []

    for start in range(0, row_count, shard_row_count):
        end = min(start + shard_row_count, row_count)
        shard = _slice_raw_pmc(raw_pmc_df, start, end)
        shard_views.append(shard)

    return shard_views


def evaluate_expression_on_shard(
    shard_view: Union[pd.DataFrame, dict],
    expr: str,
    eval_context_template: dict[str, object],
) -> object:
    """Evaluate an expression against a single shard.

    Overrides raw_pmc_df in the template context with
    the shard's data before evaluation.
    """
    context = {
        **eval_context_template,
        "raw_pmc_df": shard_view,
    }
    return eval(
        compile(expr, "<string>", "eval"),
        {},
        context,
    )


def concatenate_shard_results(
    shard_results: list,
) -> object:
    """Combine per-shard evaluation results into one value.

    Concatenates Series or ndarray results. Returns the first
    result unchanged for scalar values.
    """
    first_result = shard_results[0]
    if isinstance(first_result, pd.Series):
        return pd.concat(shard_results)
    if isinstance(first_result, np.ndarray):
        return np.concatenate(shard_results)
    return first_result


def compute_sharded_metric(
    shard_views: list[Union[pd.DataFrame, dict]],
    expr: str,
    eval_context_template: dict[str, object],
    thread_count: int,
) -> object:
    """Evaluate expression across shards in parallel.

    Submits each shard to a ThreadPoolExecutor with
    thread_count workers, then concatenates all results.
    """
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [
            executor.submit(
                evaluate_expression_on_shard,
                shard,
                expr,
                eval_context_template,
            )
            for shard in shard_views
        ]
        shard_results = [future.result() for future in futures]
    return concatenate_shard_results(shard_results)


def _extract_pmc_perf(
    raw_pmc_df: Union[pd.DataFrame, dict],
) -> Optional[pd.DataFrame]:
    """Get the pmc_perf DataFrame from raw PMC data."""
    if isinstance(raw_pmc_df, dict):
        return raw_pmc_df.get("pmc_perf")
    if isinstance(raw_pmc_df, pd.DataFrame):
        pmc_perf = raw_pmc_df.get("pmc_perf")
        if isinstance(pmc_perf, pd.DataFrame):
            return pmc_perf
    return None


def _slice_raw_pmc(
    raw_pmc_df: Union[pd.DataFrame, dict],
    start: int,
    end: int,
) -> Union[pd.DataFrame, dict]:
    """Create a row-sliced view of raw PMC data."""
    if isinstance(raw_pmc_df, dict):
        shard: dict = {}
        for key, value in raw_pmc_df.items():
            if isinstance(value, pd.DataFrame):
                shard[key] = value.iloc[start:end]
            else:
                shard[key] = value
        return shard
    return raw_pmc_df.iloc[start:end]
