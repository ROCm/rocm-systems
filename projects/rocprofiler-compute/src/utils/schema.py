# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from membw_analysis.models import MemBwAnalysisResult


@dataclass
class ArchConfig:
    # [id: panel_config] pairs
    panel_configs: OrderedDict[int, Any] = field(default_factory=OrderedDict)

    # [id: df] pairs
    dfs: dict[int, pd.DataFrame] = field(default_factory=dict)

    # NB:
    #  dfs_type should be a meta info embeded into df.
    #  pandas.DataFrame.attrs is experimental and may change without warning.
    #  So do it as below for now.

    # [id: df_type] pairs
    dfs_type: dict[int, str] = field(default_factory=dict)

    # [id: list of formula strings] pairs
    dfs_expressions: dict[int, list[str]] = field(default_factory=dict)

    # [Index: Metric name] pairs
    metric_list: dict[str, str] = field(default_factory=dict)

    # [Metric name: Counters] pairs
    metric_counters: dict[str, list] = field(default_factory=dict)


@dataclass
class MlApiTracePair:
    """Marker rows from one profiling pass and the sibling counter CSV."""

    marker_df: pd.DataFrame
    counter_path: Path
    joined_df: Optional[pd.DataFrame] = None


@dataclass
class Workload:
    sys_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_pmc: pd.DataFrame = field(default_factory=pd.DataFrame)
    dfs: dict[int, pd.DataFrame] = field(default_factory=dict)
    dfs_type: dict[int, str] = field(default_factory=dict)
    filter_kernel_ids: list[int] = field(default_factory=list)
    filter_gpu_ids: list[int] = field(default_factory=list)
    filter_dispatch_ids: list[int] = field(default_factory=list)
    avail_ips: list[int] = field(default_factory=list)
    roofline_peaks: pd.DataFrame = field(default_factory=pd.DataFrame)
    roofline_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)
    path: str = field(default_factory=str)
    filter_top_n: str = field(default_factory=str)
    # Marker CSV / counter CSV pairs, one entry per profiling pass.
    ml_api_trace_pairs: list[MlApiTracePair] = field(default_factory=list)
    # Dispatches whose Correlation_ID is missing from that pass's marker CSV.
    unmatched_kernel_frames: list[pd.DataFrame] = field(default_factory=list)
    # Consolidated marker rows after matching operator calls across passes.
    ml_api_trace_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Nested operator trees keyed by Thread_Id.
    ml_api_call_trees: dict[str, list[Any]] = field(default_factory=dict)
    # Glob-matched operator nodes from --torch-operator / --triton-operator.
    ml_api_glob_matches: list[Any] = field(default_factory=list)
    membw_result: Optional[MemBwAnalysisResult] = None


# Stem of the merged counter intermediate; csv_compression owns the suffix.
PMC_PERF_FILE_PREFIX = "pmc_perf"
