#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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


def test_agent_info(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    gpu_count = 0
    for agent in data["agents"]:
        # type 1 = CPU, type 2 = GPU
        assert agent["type"] in (1, 2)
        if agent["type"] == 1:
            assert agent["cpu_cores_count"] > 0
            assert agent["simd_count"] == 0
            assert agent["max_waves_per_simd"] == 0
        else:
            gpu_count += 1
            assert agent["cpu_cores_count"] == 0
            assert agent["simd_count"] > 0
            assert agent["max_waves_per_simd"] > 0

    assert gpu_count > 0, "No GPU agents found"


def test_kernel_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    assert (
        len(kernel_dispatch_data) > 0
    ), "No kernel dispatches captured during attachment"

    simple_kernel_found = False
    kernel_threads = set()

    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])

        assert get_kind_name(dispatch["kind"]) == "KERNEL_DISPATCH"
        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]
        assert dispatch_info["queue_id"]["handle"] > 0

        if "simple_kernel" in kernel_name:
            simple_kernel_found = True
            assert dispatch_info["workgroup_size"]["x"] == 256
            assert dispatch_info["workgroup_size"]["y"] == 1
            assert dispatch_info["workgroup_size"]["z"] == 1
            assert dispatch_info["grid_size"]["x"] >= 1
            assert dispatch_info["grid_size"]["y"] >= 1
            assert dispatch_info["grid_size"]["z"] >= 1
            kernel_threads.add(dispatch["thread_id"])

    assert simple_kernel_found, "Expected 'simple_kernel' not found in kernel dispatches"
    assert (
        len(kernel_threads) == 8
    ), f"Expected 8 unique threads, got {len(kernel_threads)}"


def test_memory_copy_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def get_agent(agent_id):
        for agent in data["agents"]:
            if agent["id"]["handle"] == agent_id["handle"]:
                return agent
        return None

    memory_copy_data = data["buffer_records"]["memory_copy"]
    assert (
        len(memory_copy_data) > 0
    ), "No memory copy operations captured during attachment"

    host_to_device_count = 0
    device_to_host_count = 0

    for record in memory_copy_data:
        assert get_kind_name(record["kind"]) == "MEMORY_COPY"
        assert record["correlation_id"]["internal"] > 0
        assert record["end_timestamp"] >= record["start_timestamp"]

        src_agent = get_agent(record["src_agent_id"])
        dst_agent = get_agent(record["dst_agent_id"])
        assert src_agent is not None, f"src agent not found: {record['src_agent_id']}"
        assert dst_agent is not None, f"dst agent not found: {record['dst_agent_id']}"

        # type 1 = CPU, type 2 = GPU
        if src_agent["type"] == 1 and dst_agent["type"] == 2:
            host_to_device_count += 1
        elif src_agent["type"] == 2 and dst_agent["type"] == 1:
            device_to_host_count += 1

    assert host_to_device_count > 0, "No host-to-device memory copies captured"
    assert device_to_host_count > 0, "No device-to-host memory copies captured"


def test_hsa_api_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def get_operation_name(kind_id, op_id):
        return data["strings"]["buffer_records"][kind_id]["operations"][op_id]

    valid_domains = (
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HSA_IMAGE_EXT_API",
        "HSA_FINALIZE_EXT_API",
    )

    hsa_api_data = data["buffer_records"]["hsa_api"]
    assert len(hsa_api_data) > 0, "No HSA API calls captured during attachment"

    functions = []
    for record in hsa_api_data:
        kind = get_kind_name(record["kind"])
        assert kind in valid_domains, f"Unexpected HSA domain: {kind}"
        assert record["thread_id"] > 0
        assert record["end_timestamp"] >= record["start_timestamp"]
        functions.append(get_operation_name(record["kind"], record["operation"]))

    assert any(
        "memory" in f.lower() for f in functions
    ), "No memory-related HSA functions captured"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
