#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import pytest
import re
import os

from collections import Counter


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


def extract_number(pattern, string):
    match = re.match(pattern, string)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Pattern '{pattern}' not found in '{string}'.")


def find_key_with_substring(data, substring):
    return next((k for k in data.keys() if substring in k), None)


def check_tot_data(tot_data):
    iteration_msg = find_key_with_substring(tot_data, "iterations:")
    assert tot_data[iteration_msg] == 1

    num_iterations = extract_number(r"iterations: (\d+)", iteration_msg)
    assert tot_data["main loop"] == num_iterations

    if num_iterations > 0:
        run_msg = find_key_with_substring(tot_data, "run")
        if run_msg is not None:
            value = extract_number(r"run\((\d+)\)", run_msg)

            assert tot_data[f"run({value})"] == num_iterations

            assert "fib" in tot_data.keys()
            assert tot_data[f"fib(n={value})"] == num_iterations
            for n in range(value, int(max([value / 2, value - 10]))):
                assert f"fib(n={n})" in tot_data.keys()

            assert tot_data[f"inefficient({value})"] == num_iterations
            assert tot_data[f"sum"] == num_iterations
            sum_msg = find_key_with_substring(tot_data, "sum(nelem=")
            assert tot_data[sum_msg] == num_iterations


def test_marker_api_trace(marker_input_data):
    functions = []

    for row in marker_input_data:
        assert row["Domain"] in [
            "MARKER_CORE_API",
            "MARKER_CONTROL_API",
            "MARKER_NAME_API",
            "MARKER_CORE_RANGE_API",
        ]
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) == 0 or int(row["Thread_Id"]) >= int(
            row["Process_Id"]
        )
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

        functions.append(row["Function"])

    check_tot_data(Counter(functions))


def test_marker_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data.strings.buffer_records[kind_id]["kind"]

    valid_domain = (
        "MARKER_CORE_API",
        "MARKER_CONTROL_API",
        "MARKER_NAME_API",
        "MARKER_CORE_RANGE_API",
    )

    marker_data = data.buffer_records.marker_api

    for marker in marker_data:
        assert get_kind_name(marker["kind"]) in valid_domain
        assert marker["thread_id"] >= data["metadata"]["pid"]
        assert marker["end_timestamp"] >= marker["start_timestamp"]

    tot_data = Counter([m["value"] for m in data.strings.marker_api])
    check_tot_data(tot_data)


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
