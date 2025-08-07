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

import re
import sys
import pytest


class dim3(object):
    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def as_tuple(self):
        return (self.x, self.y, self.z)


def validate_average(row):
    avg_ns = float(row["AverageNs"])
    tot_ns = int(row["TotalDurationNs"])
    num_cnt = float(row["Calls"])
    assert abs(avg_ns - (tot_ns / num_cnt)) < 1.0e-3, f"{row}"


def validate_stats(row):
    min_v = int(row["MinNs"])
    max_v = int(row["MaxNs"])
    avg_v = float(row["AverageNs"])
    cnt_v = int(row["Calls"])
    stddev_v = float(row["StdDev"])

    assert min_v > 0, f"{row}"
    assert max_v > 0, f"{row}"
    assert min_v < max_v if cnt_v > 1 else min_v == max_v, f"{row}"
    assert min_v < avg_v if cnt_v > 1 else min_v == int(avg_v), f"{row}"
    assert max_v > avg_v if cnt_v > 1 else max_v == int(avg_v), f"{row}"
    assert stddev_v > 0.0 if cnt_v > 1 else int(stddev_v) == 0, f"{row}"


def test_api_trace(
    hsa_input_data,
    hip_input_data,
    marker_input_data,
    kernel_input_data,
    memory_copy_input_data,
    hip_stats_data,
):
    functions = []
    hsa_correlation_ids = []
    hip_correlation_ids = []
    marker_correlation_ids = []
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
        cid = int(row["Correlation_Id"])
        hsa_correlation_ids.append(cid)

    for row in hip_input_data:
        assert row["Domain"] in [
            "HIP_RUNTIME_API",
            "HIP_COMPILER_API",
            "HIP_RUNTIME_API_EXT",
            "HIP_COMPILER_API_EXT",
        ]
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) == 0 or int(row["Thread_Id"]) >= int(
            row["Process_Id"]
        )
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        cid = int(row["Correlation_Id"])
        hip_correlation_ids.append(cid)

    for row in marker_input_data:
        assert row["Domain"] in ["MARKER_CORE_API", "MARKER_CORE_RANGE_API"]
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) == 0 or int(row["Thread_Id"]) >= int(
            row["Process_Id"]
        )
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        cid = int(row["Correlation_Id"])
        # Correlation ID will be identical for MARKER_CORE_API and MARKER_CORE_RANGE_API
        marker_correlation_ids.append(cid)

    def get_sorted_unique(inp):
        return sorted(list(set(inp)))

    def diagnose_non_unique(_input_data):
        _corr_id_hist = {}
        for row in _input_data:
            _cid = int(row["Correlation_Id"])
            # ensure duplicate does not already exist
            assert (
                _cid not in _corr_id_hist.keys()
            ), f"\ncurrent : {row}\nprevious: {_corr_id_hist[_cid]}"
            _corr_id_hist[_cid] = row

    if len(hsa_correlation_ids) != len(get_sorted_unique(hsa_correlation_ids)):
        diagnose_non_unique(hsa_input_data)

    if len(hip_correlation_ids) != len(get_sorted_unique(hip_correlation_ids)):
        diagnose_non_unique(hip_input_data)

    correlation_ids = get_sorted_unique(
        hsa_correlation_ids + hip_correlation_ids + marker_correlation_ids
    )

    # make sure that we have associated API calls for all async ops
    for itr in [kernel_input_data, memory_copy_input_data]:
        for row in itr:
            cid = int(row["Correlation_Id"])
            assert (
                cid in correlation_ids
            ), f"[{cid}] {row}\nCorrelation IDs:\n\t{correlation_ids}"

    # all correlation ids are unique
    if len(correlation_ids) != (len(hsa_input_data) + len(hip_input_data)):
        for itr in hsa_input_data:
            assert int(itr["Correlation_Id"]) in correlation_ids, f"{itr}"
        for itr in hip_input_data:
            assert int(itr["Correlation_Id"]) in correlation_ids, f"{itr}"

    assert len(correlation_ids) == (
        len(hsa_input_data) + len(hip_input_data) + len(marker_input_data)
    )
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == 1, f"{correlation_ids}"

    # disable below because this is unstable. Need to diagnose why this is unstable
    # assert correlation_ids[-1] == len(correlation_ids) + 5, f"{correlation_ids}"

    functions = list(set(functions))
    for itr in (
        "hsa_amd_memory_async_copy_on_engine",
        "hsa_agent_get_info",
        "hsa_agent_iterate_isas",
        "hsa_signal_create",
        "hsa_agent_get_info",
        "hsa_executable_symbol_get_info",
    ):
        assert itr in functions
    if hip_input_data:
        for itr in (
            "hipGetLastError",
            "hipLaunchKernel",
            "hipStreamSynchronize",
            "hipMemcpyAsync",
            "hipFree",
            "hipStreamDestroy",
            "hipDeviceSynchronize",
            "hipDeviceReset",
            "hipSetDevice",
        ):
            assert itr in functions

    for row in hip_stats_data:
        assert int(row["TotalDurationNs"]) > 0
        assert int(row["Calls"]) > 0
        validate_average(row)
        assert float(row["Percentage"]) > 0.0
        validate_stats(row)


def test_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    metadata = data["metadata"]
    names = data["strings"]["buffer_records"]
    buffer_records = data["buffer_records"]
    hsa_data = buffer_records["hsa_api"]
    hip_data = buffer_records["hip_api"]
    marker_data = buffer_records["marker_api"]

    valid_domain = [
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HSA_IMAGE_EXT_API",
        "HSA_FINALIZE_EXT_API",
    ]

    valid_hip_domain = [
        "HIP_RUNTIME_API",
        "HIP_COMPILER_API",
        "HIP_RUNTIME_API_EXT",
        "HIP_COMPILER_API_EXT",
    ]

    valid_marker_domain = [
        "MARKER_CORE_API",
        "MARKER_CORE_RANGE_API",
    ]

    def get_operation_name(kind_id, op_id):
        return names[kind_id]["operations"][op_id]

    def get_kind_name(kind_id):
        return names[kind_id]["kind"]

    assert metadata["pid"] > 0

    functions = []
    correlation_ids = []
    for api in hsa_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_domain
        assert api["thread_id"] >= metadata["pid"]
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))
        correlation_ids.append(api["correlation_id"]["internal"])

    for api in hip_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_hip_domain
        assert metadata["pid"] > 0
        assert api["thread_id"] == 0 or api["thread_id"] >= metadata["pid"]
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))
        correlation_ids.append(api["correlation_id"]["internal"])

    for api in marker_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_marker_domain
        assert metadata["pid"] > 0
        assert api["thread_id"] == 0 or api["thread_id"] >= metadata["pid"]
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))
        correlation_ids.append(api["correlation_id"]["internal"])

    correlation_ids = sorted(list(set(correlation_ids)))

    # all correlation ids are unique
    assert len(correlation_ids) == (len(hsa_data) + len(hip_data) + len(marker_data))
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == 1, f"{correlation_ids}"

    # disable below because this is unstable. Need to diagnose why this is unstable
    # assert correlation_ids[-1] == len(correlation_ids) + 5, f"{correlation_ids}"

    functions = list(set(functions))
    for itr in (
        "hsa_amd_memory_async_copy_on_engine",
        "hsa_agent_get_info",
        "hsa_agent_iterate_isas",
        "hsa_signal_create",
        "hsa_agent_get_info",
        "hsa_executable_symbol_get_info",
    ):
        assert itr in functions
    if hip_data:
        for itr in (
            "hipGetLastError",
            "hipLaunchKernel",
            "hipStreamSynchronize",
            "hipMemcpyAsync",
            "hipFree",
            "hipStreamDestroy",
            "hipDeviceSynchronize",
            "hipDeviceReset",
            "hipSetDevice",
        ):
            assert itr in functions


def test_kernel_trace(kernel_input_data, kernel_stats_data):
    valid_kernel_names = sorted(
        [
            "(anonymous namespace)::transpose(int const*, int*, int, int)",
            "void (anonymous namespace)::addition_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::divide_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::multiply_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::subtract_kernel<float>(float*, float const*, float const*, int, int)",
        ]
    )

    kernels = []
    for row in kernel_input_data:
        kernel_name = row["Kernel_Name"]
        if re.search(r"__amd_rocclr_.*", kernel_name):
            continue

        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"].split(" ")[-1]) >= 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert int(row["Correlation_Id"]) > 0
        assert kernel_name in valid_kernel_names

        if kernel_name not in kernels:
            kernels.append(kernel_name)

        workgrp_size = dim3(
            row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"]
        )
        grid_size = dim3(row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"])

        if kernel_name == "__amd_rocclr_fillBufferAligned":
            assert workgrp_size.as_tuple() > (1, 1, 1)
            assert grid_size.as_tuple() > (1, 1, 1)
        elif "transpose" in kernel_name:
            assert workgrp_size.as_tuple() == (32, 32, 1)
            assert grid_size.as_tuple() == (9920, 9920, 1)
        else:
            assert workgrp_size.as_tuple() == (64, 1, 1)
            assert grid_size.as_tuple() == (4096, 2048, 1)

        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    kernels = sorted(list(set(kernels)))
    assert kernels == valid_kernel_names

    for row in kernel_stats_data:
        assert int(row["TotalDurationNs"]) > 0
        assert int(row["Calls"]) > 0
        validate_average(row)
        assert float(row["Percentage"]) > 0.0
        validate_stats(row)


def test_kernel_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    buffer_records = data["buffer_records"]
    names = data["strings"]["buffer_records"]

    valid_kernel_names = sorted(
        [
            "(anonymous namespace)::transpose(int const*, int*, int, int)",
            "void (anonymous namespace)::addition_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::divide_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::multiply_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::subtract_kernel<float>(float*, float const*, float const*, int, int)",
        ]
    )

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    kernels = []
    for row in buffer_records["kernel_dispatch"]:
        dispatch_info = row["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        if re.search(r"__amd_rocclr_.*", kernel_name):
            continue

        kernels.append(kernel_name)

        assert names[row["kind"]]["kind"] == "KERNEL_DISPATCH"
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["kernel_id"] > 0
        assert row["correlation_id"]["internal"] > 0
        assert kernel_name in valid_kernel_names, f"row:\n\t{row}"

        workgrp_size = dim3(
            dispatch_info["workgroup_size"]["x"],
            dispatch_info["workgroup_size"]["y"],
            dispatch_info["workgroup_size"]["z"],
        )
        grid_size = dim3(
            dispatch_info["grid_size"]["x"],
            dispatch_info["grid_size"]["y"],
            dispatch_info["grid_size"]["z"],
        )

        if kernel_name == "__amd_rocclr_fillBufferAligned":
            assert workgrp_size.as_tuple() > (1, 1, 1)
            assert grid_size.as_tuple() > (1, 1, 1)
        elif "transpose" in kernel_name:
            assert workgrp_size.as_tuple() == (32, 32, 1)
            assert grid_size.as_tuple() == (9920, 9920, 1)
        else:
            assert workgrp_size.as_tuple() == (64, 1, 1)
            assert grid_size.as_tuple() == (4096, 2048, 1)

        assert int(row["end_timestamp"]) >= int(row["start_timestamp"])

    kernels = sorted(list(set(kernels)))
    assert kernels == valid_kernel_names


def test_memory_copy_trace(
    agent_info_input_data,
    memory_copy_input_data,
    hsa_input_data,
    hsa_stats_data,
    memory_copy_stats_data,
):
    def get_agent(node_id):
        for row in agent_info_input_data:
            if row["Logical_Node_Id"] == node_id:
                return row
        return None

    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"
        assert row["Direction"] in (
            "MEMORY_COPY_HOST_TO_DEVICE",
            "MEMORY_COPY_DEVICE_TO_HOST",
        )

        src_agent = get_agent(row["Source_Agent_Id"].split(" ")[-1])
        dst_agent = get_agent(row["Destination_Agent_Id"].split(" ")[-1])
        assert src_agent is not None and dst_agent is not None, f"{agent_info_input_data}"

        if row["Direction"] == "MEMORY_COPY_HOST_TO_DEVICE":
            assert src_agent["Agent_Type"] == "CPU"
            assert dst_agent["Agent_Type"] == "GPU"
        elif row["Direction"] == "MEMORY_COPY_DEVICE_TO_HOST":
            assert src_agent["Agent_Type"] == "GPU"
            assert dst_agent["Agent_Type"] == "CPU"

        assert int(row["Correlation_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    valid_length = 0
    for row in hsa_input_data:
        if re.search(r".*memory_async_copy.*", row["Function"]):
            valid_length += 1
    assert len(memory_copy_input_data) == valid_length

    for row in hsa_stats_data:
        assert int(row["TotalDurationNs"]) > 0
        assert int(row["Calls"]) > 0
        validate_average(row)
        assert float(row["Percentage"]) > 0.0
        validate_stats(row)

    for row in memory_copy_stats_data:
        assert int(row["TotalDurationNs"]) > 0
        assert int(row["Calls"]) > 0
        validate_average(row)
        assert float(row["Percentage"]) > 0.0
        validate_stats(row)


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    # do not test for HSA and HIP since that may vary slightly b/t two separate runs
    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("marker", "kernel", "memory_copy")
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    # do not test for HSA and HIP since that may vary slightly b/t two separate runs
    rocprofv3.test_otf2_data(otf2_data, json_data, ("marker", "kernel", "memory_copy"))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
