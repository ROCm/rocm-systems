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
    counter_groups = [{"SQ_WAVES": 0, "GRBM_COUNT": 0}, {"GRBM_GUI_ACTIVE": 0}]
    di_list = []

    for row in counter_input_data:
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0

        assert len(row["Counter_Value"]) > 0
        # assert row["Counter_Name"].contains("SQ_WAVES").all()
        assert row["Counter_Name"] in counter_names
        assert float(row["Counter_Value"]) > 0
        row_id = (int(row["Dispatch_Id"]) - 1) % 2
        assert row["Counter_Name"] in counter_groups[int(row_id)]
        counter_groups[int(row_id)][row["Counter_Name"]] += 1
        di_list.append(int(row["Dispatch_Id"]))

    for group in counter_groups:
        for counter in group:
            assert group[counter] > 0, f"Counter {counter} not found in the group"

    # # make sure the dispatch ids are unique and ordered
    di_list = list(dict.fromkeys(di_list))
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_list


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
