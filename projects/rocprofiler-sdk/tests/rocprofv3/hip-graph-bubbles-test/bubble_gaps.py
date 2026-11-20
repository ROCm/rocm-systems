#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""
Bubble analysis of the kernel trace of the hip-graph-bubbles test.

validate.py and run_with_bubble_retry.py both use this module, so that the retry
decision and the validation apply the same criterion.
"""

import os
from collections import defaultdict

# The typing aliases, not list[...] or tuple[...]: the RHEL 8 and SLES 15 CI images run
# these tests with Python 3.6.
from typing import Dict, List, Tuple

Row = Dict[str, str]

# WSL/DXG does not expose /dev/kfd, so kernel dispatches are scheduled through the
# host /dev/dxg path. The resulting inter-dispatch gaps reflect host scheduler
# delays (hundreds of microseconds to tens of milliseconds), not a profiler
# regression, so the bubble-timing check is only meaningful on real KFD HW.
KFD_AVAILABLE = os.path.exists("/dev/kfd")

# Check for batching patterns (most gaps are small, but some are very large). This
# detects scenarios where kernels are grouped into batches with large gaps between
# batches, indicating a profiler regression.
OUTLIER_THRESHOLD_NS = 50_000  # 50 microseconds


def select_graph_kernels(rows: List[Row]) -> List[Row]:
    """Keep only the simpleKernel and emptyKernel dispatches (exclude BLIT kernels)."""
    ret_arr = [
        row
        for row in rows
        if "simpleKernel" in row.get("Kernel_Name", "")
        or "emptyKernel" in row.get("Kernel_Name", "")
    ]
    return ret_arr


def group_by_launch(rows: List[Row]) -> Dict[int, List[Row]]:
    """Group dispatches by Correlation_Id (each ID is one hipGraphLaunch)."""
    ret_arr = defaultdict(list)
    for row in rows:
        try:
            ret_arr[int(row["Correlation_Id"])].append(row)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse Correlation_Id from row: {row}") from e
    return ret_arr


def launch_gaps(iterations: Dict[int, List[Row]]) -> List[int]:
    """Return the gaps in ns between consecutive dispatches of each graph launch."""
    ret_arr = []
    for corr_id, dispatches in iterations.items():
        # Sort by start timestamp within this graph launch
        sorted_dispatches = sorted(dispatches, key=lambda r: int(r["Start_Timestamp"]))

        # Correlation IDs may span multiple or partial graph launches, so the number
        # of dispatches per correlation ID is not checked
        for i in range(len(sorted_dispatches) - 1):
            try:
                curr_end = int(sorted_dispatches[i]["End_Timestamp"])
                next_start = int(sorted_dispatches[i + 1]["Start_Timestamp"])
                ret_arr.append(next_start - curr_end)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Correlation ID {corr_id}, dispatch pair {i},{i + 1}: "
                    f"failed to parse timestamps from {sorted_dispatches[i]} and "
                    f"{sorted_dispatches[i + 1]}"
                ) from e
    return ret_arr


def outlier_gaps(gaps: List[int]) -> Tuple[int, int]:
    """Return the 99.9th percentile gap and the maximum gap in ns."""
    gaps_sorted = sorted(gaps)
    ret = (gaps_sorted[int(len(gaps_sorted) * 0.999)], gaps_sorted[-1])
    return ret
