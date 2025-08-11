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

kernel_trace_list = sorted(["addition_kernel", "subtract_kernel"])
kernel_counter_list = ["addition_kernel"]


def unique(lst):
    return list(set(lst))


def test_counter_collection_json_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    counter_collection_data = data["callback_records"]["counter_collection"]

    for counter in counter_collection_data:
        kernel_name = get_kernel_name(counter.dispatch_data.dispatch_info.kernel_id)
        assert kernel_name in kernel_counter_list


def test_kernel_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    kernels = []
    assert len(kernel_dispatch_data) == 2
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])

        assert get_kind_name(dispatch["kind"]) == "KERNEL_DISPATCH"
        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["kernel_id"] > 0
        if not re.search(r"__amd_rocclr_.*", kernel_name):
            kernels.append(kernel_name)

        assert dispatch_info["workgroup_size"]["x"] == 64
        assert dispatch_info["workgroup_size"]["y"] == 1
        assert dispatch_info["workgroup_size"]["z"] == 1
        assert dispatch_info["grid_size"]["x"] == 1024
        assert dispatch_info["grid_size"]["y"] == 1024
        assert dispatch_info["grid_size"]["z"] == 1
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]

    assert kernels == kernel_trace_list


def test_hip_api_trace_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_operation_name(kind_id, op_id):
        return data["strings"]["buffer_records"][kind_id]["operations"][op_id]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_domain_names = (
        "HIP_RUNTIME_API",
        "HIP_RUNTIME_API_EXT",
    )

    hip_api_data = data["buffer_records"]["hip_api"]

    functions = []
    for api in hip_api_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_domain_names
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))

    expected_functions = (
        [
            "hipGetDeviceCount",
            "hipSetDevice",
            "hipDeviceSynchronize",
            "hipStreamCreateWithFlags",
        ]
        + (["hipHostMalloc"] * 3)
        + (["hipMallocAsync"] * 3)
        + (["hipMemcpyAsync"] * 2)
        + [
            "hipStreamSynchronize",
            "hipDeviceSynchronize",
            "hipLaunchKernel",
            "hipGetLastError",
            "hipLaunchKernel",
            "hipGetLastError",
        ]
    )
    assert functions == expected_functions


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
