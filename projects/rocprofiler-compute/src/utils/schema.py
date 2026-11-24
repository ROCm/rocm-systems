# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from collections import ChainMap, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from membw_analysis.models import MemBwAnalysisResult
from ml_api_tracing.analyze.definitions import CallTreeForest


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


@dataclass
class Workload:
    sys_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_pmc: pd.DataFrame = field(default_factory=pd.DataFrame)
    # One counter table per pass, on raw_pmc's rows, keyed by pass name.
    pass_tables: Dict[str, ChainMap] = field(default_factory=dict)
    dfs: dict[int, pd.DataFrame] = field(default_factory=dict)
    dfs_type: dict[int, str] = field(default_factory=dict)
    filter_kernel_ids: list[int] = field(default_factory=list)
    filter_gpu_ids: list[int] = field(default_factory=list)
    filter_dispatch_ids: list[int] = field(default_factory=list)
    roofline_peaks: pd.DataFrame = field(default_factory=pd.DataFrame)
    roofline_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)
    roofline_html_path: Optional[str] = None
    path: str = field(default_factory=str)
    # Consolidated marker rows after matching operator calls across passes.
    ml_api_trace_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Nested call-tree roots keyed by Thread_Id.
    ml_api_call_trees: CallTreeForest = field(default_factory=dict)
    # --operator patterns keyed by the backend they match in this workload.
    operator_patterns: Dict[str, List[str]] = field(default_factory=dict)
    # Kernel name to the id shown in the top-kernel table, for operator reports.
    operator_kernel_ids: Dict[str, int] = field(default_factory=dict)
    membw_result: Optional[MemBwAnalysisResult] = None
