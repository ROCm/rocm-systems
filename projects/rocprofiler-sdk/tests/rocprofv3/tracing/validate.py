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
import re


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


def test_hsa_api_trace(hsa_input_data):
    functions = []
    correlation_ids = []
    for row in hsa_input_data:
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

    hsa_api_calls_offset = 2  # roctxRangePush is first
    num_marker_api_calls = 6  # seven marker API calls, only six entries in
    # marker csv data because roctxRangePush + roctxRangePop is one entry

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_input_data)
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == hsa_api_calls_offset
    assert correlation_ids[-1] == len(correlation_ids) + num_marker_api_calls

    functions = list(set(functions))
    assert "hsa_amd_memory_async_copy_on_engine" in functions


def test_hsa_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_operation_name(kind_id, op_id):
        return data["strings"]["buffer_records"][kind_id]["operations"][op_id]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_domain_names = (
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HSA_IMAGE_EXT_API",
        "HSA_FINALIZE_EXT_API",
    )

    hsa_api_data = data["buffer_records"]["hsa_api"]

    functions = []
    correlation_ids = []
    for api in hsa_api_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_domain_names
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))
        correlation_ids.append(api["correlation_id"]["internal"])

    correlation_ids = sorted(list(set(correlation_ids)))

    hsa_api_calls_offset = 2  # roctxRangePush is first
    num_marker_api_calls = 6  # seven marker API calls, only six entries in
    # marker csv data because roctxRangePush + roctxRangePop is one entry

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_api_data)
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == hsa_api_calls_offset
    assert correlation_ids[-1] == len(correlation_ids) + num_marker_api_calls

    functions = list(set(functions))
    assert "hsa_amd_memory_async_copy_on_engine" in functions


def test_kernel_trace(kernel_input_data):
    valid_kernel_names = (
        "_Z15matrixTransposePfS_i.kd",
        "matrixTranspose(float*, float*, int)",
    )

    assert len(kernel_input_data) == 1
    for row in kernel_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"].split(" ")[-1]) >= 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert row["Kernel_Name"] in valid_kernel_names
        assert int(row["Correlation_Id"]) > 0
        assert int(row["Workgroup_Size_X"]) == 4
        assert int(row["Workgroup_Size_Y"]) == 4
        assert int(row["Workgroup_Size_Z"]) == 1
        assert int(row["Grid_Size_X"]) == 1024
        assert int(row["Grid_Size_Y"]) == 1024
        assert int(row["Grid_Size_Z"]) == 1
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


def test_host_functions_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["truncated_kernel_name"]

    def get_host_function_name(host_function_id):
        return data["host_functions"][host_function_id]["truncated_host_function_name"]

    host_function_data = data.host_functions
    kernel_symbols_data = data.kernel_symbols
    code_objects_data = data.code_objects
    assert len(host_function_data) > 0
    for host_function in host_function_data:
        if host_function.host_function_id == 0:
            continue
        assert host_function.host_function_id > 0
        assert host_function.kernel_id > 0 and host_function.kernel_id <= len(
            kernel_symbols_data
        )
        assert host_function.code_object_id > 0 and host_function.code_object_id <= len(
            code_objects_data
        )
        assert get_host_function_name(
            host_function["host_function_id"]
        ) == get_kernel_name(host_function.kernel_id)


def test_kernel_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_kernel_names = (
        "_Z15matrixTransposePfS_i.kd",
        "matrixTranspose(float*, float*, int)",
    )
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    assert len(kernel_dispatch_data) == 1
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])

        assert get_kind_name(dispatch["kind"]) == "KERNEL_DISPATCH"
        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["kernel_id"] > 0
        if not re.search(r"__amd_rocclr_.*", kernel_name):
            assert kernel_name in valid_kernel_names

        assert dispatch_info["workgroup_size"]["x"] == 4
        assert dispatch_info["workgroup_size"]["y"] == 4
        assert dispatch_info["workgroup_size"]["z"] == 1
        assert dispatch_info["grid_size"]["x"] == 1024
        assert dispatch_info["grid_size"]["y"] == 1024
        assert dispatch_info["grid_size"]["z"] == 1
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]


def test_memory_copy_trace(agent_info_input_data, memory_copy_input_data):
    def get_agent(node_id):
        for row in agent_info_input_data:
            if row["Logical_Node_Id"] == node_id:
                return row
        return None

    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"

    assert len(memory_copy_input_data) == 2

    def test_row(idx, direction):
        assert direction in ("MEMORY_COPY_HOST_TO_DEVICE", "MEMORY_COPY_DEVICE_TO_HOST")
        row = memory_copy_input_data[idx]
        assert row["Direction"] == direction
        src_agent = get_agent(row["Source_Agent_Id"].split(" ")[-1])
        dst_agent = get_agent(row["Destination_Agent_Id"].split(" ")[-1])
        assert src_agent is not None and dst_agent is not None, f"{agent_info_input_data}"
        if direction == "MEMORY_COPY_HOST_TO_DEVICE":
            assert src_agent["Agent_Type"] == "CPU"
            assert dst_agent["Agent_Type"] == "GPU"
        else:
            assert src_agent["Agent_Type"] == "GPU"
            assert dst_agent["Agent_Type"] == "CPU"
        assert int(row["Correlation_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    test_row(0, "MEMORY_COPY_HOST_TO_DEVICE")
    test_row(1, "MEMORY_COPY_DEVICE_TO_HOST")


def test_memory_copy_json_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    buffer_records = data["buffer_records"]
    agent_data = data["agents"]
    memory_copy_data = buffer_records["memory_copy"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def get_agent(node_id):
        for agent in agent_data:
            if agent["id"]["handle"] == node_id["handle"]:
                return agent
        return None

    # one threads * two directions
    assert len(memory_copy_data) >= 2, f"{memory_copy_data}"
    assert (len(memory_copy_data) % 2) == 0, f"{memory_copy_data}"

    def test_row(idx, direction):
        assert direction in ("MEMORY_COPY_HOST_TO_DEVICE", "MEMORY_COPY_DEVICE_TO_HOST")
        row = memory_copy_data[idx]
        src_agent = get_agent(row["src_agent_id"])
        dst_agent = get_agent(row["dst_agent_id"])
        assert get_kind_name(row["kind"]) == "MEMORY_COPY"
        assert src_agent is not None, f"{row}"
        assert dst_agent is not None, f"{row}"
        if direction == "MEMORY_COPY_HOST_TO_DEVICE":
            assert src_agent["type"] == 1
            assert dst_agent["type"] == 2
        else:
            assert src_agent["type"] == 2
            assert dst_agent["type"] == 1
        assert row["correlation_id"]["internal"] > 0
        assert row["end_timestamp"] >= row["start_timestamp"]

    test_row(0, "MEMORY_COPY_HOST_TO_DEVICE")
    test_row(1, "MEMORY_COPY_DEVICE_TO_HOST")


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

    functions = list(set(functions))
    assert "main" in functions


def test_marker_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_domain = (
        "MARKER_CORE_API",
        "MARKER_CONTROL_API",
        "MARKER_NAME_API",
        "MARKER_CORE_RANGE_API",
    )

    buffer_records = data["buffer_records"]
    marker_data = buffer_records["marker_api"]
    for marker in marker_data:
        assert get_kind_name(marker["kind"]) in valid_domain
        assert marker["thread_id"] >= data["metadata"]["pid"]
        assert marker["end_timestamp"] >= marker["start_timestamp"]


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


def test_rocpd_data(rocpd_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_rocpd_data(
        rocpd_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
