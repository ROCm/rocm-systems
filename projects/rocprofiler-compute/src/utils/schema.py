# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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

    # [Index: Metric name] pairs
    metric_list: dict[str, str] = field(default_factory=dict)

    # [Metric name: Counters] pairs
    metric_counters: dict[str, list] = field(default_factory=dict)


@dataclass
class Workload:
    sys_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_pmc: pd.DataFrame = field(default_factory=pd.DataFrame)
    dfs: dict[int, pd.DataFrame] = field(default_factory=dict)
    dfs_type: dict[int, str] = field(default_factory=dict)
    filter_kernel_ids: list[int] = field(default_factory=list)
    filter_gpu_ids: list[int] = field(default_factory=list)
    filter_dispatch_ids: list[int] = field(default_factory=list)
    filter_nodes: list[str] = field(default_factory=list)
    avail_ips: list[int] = field(default_factory=list)
    roofline_peaks: pd.DataFrame = field(default_factory=pd.DataFrame)
    roofline_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)
    path: str = field(default_factory=str)
    filter_top_n: str = field(default_factory=str)
    matched_torch_trace_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# The prefix of raw pmc_perf.csv
PMC_PERF_FILE_PREFIX = "pmc_perf"
