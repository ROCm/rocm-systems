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

import sys
import pytest


def test_hsa_api_trace(hsa_trace_input_data):
    functions = []
    correlation_ids = []
    for row in hsa_trace_input_data:
        assert row["Domain"] in (
            "HSA_CORE_API",
            "HSA_AMD_EXT_API",
            "HSA_IMAGE_EXT_API",
            "HSA_FINALIZE_EXT_API",
        )
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) >= int(row["Process_Id"])
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        correlation_ids.append(int(row["Correlation_Id"]))

    correlation_ids = sorted(list(set(correlation_ids)))

    # deterministic call counts
    num_queue_create_calls = 2
    num_queue_destroy_calls = 2
    num_hsa_mem_free_calls = 2

    # signal create/destroy calls
    # although the app explicitly only creates 3 signals
    # but hsa_init() internally calls hsa_signal_create
    num_hsa_signal_create_calls = 4
    num_hsa_signal_destroy_calls = 4

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_trace_input_data)

    functions = list(functions)
    assert "hsa_shut_down" in functions
    assert functions.count("hsa_queue_create") == num_queue_create_calls
    assert functions.count("hsa_queue_destroy") == num_queue_destroy_calls
    assert functions.count("hsa_memory_free") == num_hsa_mem_free_calls
    assert functions.count("hsa_signal_create") == num_hsa_signal_create_calls
    assert functions.count("hsa_signal_destroy") == num_hsa_signal_destroy_calls


def test_kernel_trace(kernel_trace_input_data):
    valid_kernel_names = ("copyA", "copyB", "copyC")

    assert len(kernel_trace_input_data) == 3
    for row in kernel_trace_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"].split(" ")[-1]) >= 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert row["Kernel_Name"] in valid_kernel_names
        assert int(row["Correlation_Id"]) > 0
        assert int(row["Workgroup_Size_X"]) == 64
        assert int(row["Workgroup_Size_Y"]) == 1
        assert int(row["Workgroup_Size_Z"]) == 1
        assert int(row["Grid_Size_X"]) == 64
        assert int(row["Grid_Size_Y"]) == 1
        assert int(row["Grid_Size_Z"]) == 1
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


def test_kernel_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    valid_kernel_names = ["copyA", "copyB", "copyC"]
    buffer_records = data["buffer_records"]
    buffer_names = data["strings"]["buffer_records"]
    kernel_dispatch_data = buffer_records["kernel_dispatch"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    assert len(kernel_dispatch_data) == 3

    for dispatch in kernel_dispatch_data:

        assert buffer_names[dispatch["kind"]]["kind"] == "KERNEL_DISPATCH"
        dispatch_info = dispatch["dispatch_info"]
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["kernel_id"] > 0

        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        assert kernel_name in valid_kernel_names

        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch_info["workgroup_size"]["x"] == 64
        assert dispatch_info["workgroup_size"]["y"] == 1
        assert dispatch_info["workgroup_size"]["z"] == 1
        assert dispatch_info["grid_size"]["x"] == 64
        assert dispatch_info["grid_size"]["y"] == 1
        assert dispatch_info["grid_size"]["z"] == 1
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]


def test_hsa_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    functions = []
    correlation_ids = []

    def get_operation_name(kind_id, op_id):
        return data["strings"]["buffer_records"][kind_id]["operations"][op_id]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    metadata = data["metadata"]
    buffer_records = data["buffer_records"]

    valid_domain_names = (
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HSA_IMAGE_EXT_API",
        "HSA_FINALIZE_EXT_API",
    )

    assert metadata["pid"] > 0
    hsa_api_data = buffer_records["hsa_api"]

    for itr in hsa_api_data:
        kind = get_kind_name(itr["kind"])
        assert kind in valid_domain_names
        assert itr["end_timestamp"] >= itr["start_timestamp"]
        functions.append(get_operation_name(itr["kind"], itr["operation"]))
        correlation_ids.append(itr["correlation_id"]["internal"])

    correlation_ids = sorted(list(set(correlation_ids)))

    # deterministic call counts
    num_queue_create_calls = 2
    num_queue_destroy_calls = 2
    num_hsa_mem_free_calls = 2

    # signal create/destroy calls
    # although the app explicitly only creates 3 signals
    # but hsa_init() internally calls hsa_signal_create
    num_hsa_signal_create_calls = 4
    num_hsa_signal_destroy_calls = 4

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_api_data)

    functions = list(functions)
    assert "hsa_shut_down" in functions
    assert functions.count("hsa_queue_create") == num_queue_create_calls
    assert functions.count("hsa_queue_destroy") == num_queue_destroy_calls
    assert functions.count("hsa_memory_free") == num_hsa_mem_free_calls
    assert functions.count("hsa_signal_create") == num_hsa_signal_create_calls
    assert functions.count("hsa_signal_destroy") == num_hsa_signal_destroy_calls


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
