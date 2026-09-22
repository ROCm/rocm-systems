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


def test_hsa_api_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_operation_name(kind_id, op_id):
        return data["strings"]["buffer_records"][kind_id]["operations"][op_id]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_domain_names = ("HSA_CORE_API",)

    hsa_api_data = data["buffer_records"]["hsa_api"]

    functions = []
    for api in hsa_api_data:
        kind = get_kind_name(api["kind"])
        assert kind in valid_domain_names
        assert api["end_timestamp"] >= api["start_timestamp"]
        functions.append(get_operation_name(api["kind"], api["operation"]))

    functions = list(set(functions))
    assert "hsa_amd_memory_async_copy_on_engine" not in functions
    assert "hsa_signal_destroy" in functions


def test_hip_api_trace(json_data):
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

    functions = list(set(functions))
    for itr in (
        "__hipPushCallConfiguration",
        "__hipPopCallConfiguration",
        "__hipRegisterFatBinary",
        "__hipRegisterFunction",
    ):
        assert itr not in functions, f"{itr}"

    for itr in (
        "hipMallocAsync",
        "hipMemcpyAsync",
        "hipMemsetAsync",
        "hipFreeAsync",
        "hipLaunchKernel",
    ):
        assert itr in functions, f"{itr}"


def test_kernel_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    def get_kernel_rename(corr_id):
        for itr in data.strings.correlation_id.external:
            if itr.key == corr_id:
                return itr.value
        return None

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    valid_kernel_names = "|".join(
        (
            "run",
            "run/iteration",
            "run/rank-([0-9]+)/thread-([0-9]+)/device-([0-9]+)/(begin|end)",
        )
    )
    valid_kernel_regex = re.compile("^({})$".format(valid_kernel_names))

    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    for dispatch in kernel_dispatch_data:
        assert get_kind_name(dispatch["kind"]) == "KERNEL_DISPATCH"
        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch["correlation_id"]["external"] > 0

        dispatch_info = dispatch["dispatch_info"]
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["queue_id"]["handle"] > 0
        assert dispatch_info["kernel_id"] > 0
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]

        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        assert (
            re.search(valid_kernel_regex, kernel_name) is None
        ), f"kernel '{kernel_name}' matches regular expression '{valid_kernel_names}'"

        assert kernel_name not in valid_kernel_names

        external_corr_id = dispatch["correlation_id"]["external"]
        assert external_corr_id > 0

        kernel_rename = get_kernel_rename(external_corr_id)
        assert kernel_rename is not None, f"{dispatch}"
        assert kernel_rename != kernel_name, f"{dispatch}"
        assert (
            re.search(valid_kernel_regex, kernel_rename) is not None
        ), f"renamed kernel '{kernel_rename}' does not match regular expression '{valid_kernel_names}'"


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

    # two threads * two directions
    assert len(memory_copy_data) >= (2 * 2), f"{memory_copy_data}"
    assert (len(memory_copy_data) % (2 * 2)) == 0, f"{memory_copy_data}"

    for row in memory_copy_data:
        src_agent = get_agent(row["src_agent_id"])
        dst_agent = get_agent(row["dst_agent_id"])
        assert get_kind_name(row["kind"]) == "MEMORY_COPY"
        assert src_agent is not None, f"{row}"
        assert dst_agent is not None, f"{row}"
        assert row["correlation_id"]["internal"] > 0
        assert row["end_timestamp"] >= row["start_timestamp"]


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "kernel", "memory_copy")
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data, json_data, ("hip", "hsa", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
