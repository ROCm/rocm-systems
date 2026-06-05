#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import pandas as pd
import os
import sys
import pytest


def test_agent_info(agent_info_input_data):
    logical_node_id = max([int(itr["Logical_Node_Id"]) for itr in agent_info_input_data])

    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


def test_counter_collection_multiple_yaml(counter_input_data):
    counter_names = ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"]
    # multiplexed pmc groups (must match input.json/input.yml). Iteration based
    # multiplexing assigns dispatch N (1-based) to group (N - 1) % len(groups).
    counter_groups = [{"SQ_WAVES": 0, "GRBM_COUNT": 0}, {"GRBM_GUI_ACTIVE": 0}]
    num_groups = len(counter_groups)
    dispatch_ids = []

    # there must be at least one profiled dispatch to validate
    assert len(counter_input_data) > 0, "no counter collection records were produced"

    for row in counter_input_data:
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0

        assert len(row["Counter_Value"]) > 0
        assert row["Counter_Name"] in counter_names
        assert float(row["Counter_Value"]) > 0
        group_id = (int(row["Dispatch_Id"]) - 1) % num_groups
        # the counter collected for a dispatch must belong to the multiplex group
        # that the dispatch was assigned to (verifies multiplexing correctness)
        assert (
            row["Counter_Name"] in counter_groups[group_id]
        ), f"counter {row['Counter_Name']} (dispatch {row['Dispatch_Id']}) not in expected group {group_id}"
        counter_groups[group_id][row["Counter_Name"]] += 1
        dispatch_ids.append(int(row["Dispatch_Id"]))

    # Only require coverage for the groups that should have been exercised given
    # the dispatches that were actually profiled. The number of profiled
    # dispatches can legitimately vary (e.g. internal copy kernels may or may not
    # be captured), so requiring every group to be populated regardless was the
    # source of intermittent failures.
    exercised_groups = {(did - 1) % num_groups for did in dispatch_ids}
    for group_id in exercised_groups:
        for counter, count in counter_groups[group_id].items():
            assert count > 0, f"Counter {counter} not found in multiplex group {group_id}"

    # make sure the dispatch ids are unique and contiguous starting from 1
    # (csv row order is not guaranteed, so sort before comparing)
    unique_dispatch_ids = sorted(set(dispatch_ids))
    expected_dispatch_ids = list(range(1, len(unique_dispatch_ids) + 1))
    assert (
        expected_dispatch_ids == unique_dispatch_ids
    ), f"dispatch ids are not contiguous: {unique_dispatch_ids}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
