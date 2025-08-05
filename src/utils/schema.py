##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################


#
# Define all common data storage classes,
# predifned dict and global functions.
#

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class ArchConfig:
    # [id: panel_config] pairs
    panel_configs: OrderedDict = field(default=dict)

    # [id: df] pairs
    dfs: Dict[int, pd.DataFrame] = field(default_factory=dict)

    # NB:
    #  dfs_type should be a meta info embeded into df.
    #  pandas.DataFrame.attrs is experimental and may change without warning.
    #  So do it as below for now.

    # [id: df_type] pairs
    dfs_type: Dict[int, str] = field(default_factory=dict)

    # [Index: Metric name] pairs
    metric_list: Dict[str, str] = field(default_factory=dict)

    # [Metric name: Counters] pairs
    metric_counters: Dict[str, list] = field(default_factory=dict)


@dataclass
class Workload:
    sys_info: pd.DataFrame = None
    raw_pmc: pd.DataFrame = None
    dfs: Dict[int, pd.DataFrame] = field(default_factory=dict)
    dfs_type: Dict[int, str] = field(default_factory=dict)
    filter_kernel_ids: List[int] = field(default_factory=list)
    filter_gpu_ids: List[int] = field(default_factory=list)
    filter_dispatch_ids: List[int] = field(default_factory=list)
    filter_nodes: List[str] = field(default_factory=list)
    avail_ips: List[int] = field(default_factory=list)


# Metrics will be calculated ONLY when the header(key) is in below list
supported_field = [
    "Value",
    "Minimum",
    "Maximum",
    "Average",
    "Median",
    "Min",
    "Max",
    "Avg",
    "Pct of Peak",
    "Peak",
    "Count",
    "Mean",
    "Pct",
    "Std Dev",
    "Q1",
    "Q3",
    "Expression",
    # Special keywords for L2 channel
    "Channel",
    "L2 Cache Hit Rate",
    "Requests",
    "L2 Read",
    "L2 Write",
    "L2 Atomic",
    "L2-Fabric Requests",
    "L2-Fabric Read",
    "L2-Fabric Write and Atomic",
    "L2-Fabric Atomic",
    "L2 Read Req",
    "L2 Write Req",
    "L2 Atomic Req",
    "L2-Fabric Read Req",
    "L2-Fabric Write and Atomic Req",
    "L2-Fabric Atomic Req",
    "L2-Fabric Read Latency",
    "L2-Fabric Write Latency",
    "L2-Fabric Atomic Latency",
    "L2-Fabric Read Stall (PCIe)",
    "L2-Fabric Read Stall (Infinity Fabric™)",
    "L2-Fabric Read Stall (HBM)",
    "L2-Fabric Write Stall (PCIe)",
    "L2-Fabric Write Stall (Infinity Fabric™)",
    "L2-Fabric Write Stall (HBM)",
    "L2-Fabric Write Starve",
]

# The prefix of raw pmc_perf.csv
pmc_perf_file_prefix = "pmc_perf"
