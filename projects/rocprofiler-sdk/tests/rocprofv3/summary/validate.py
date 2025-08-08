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

import os
import re
import sys
import pytest


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
            print(itr)
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


def test_memory_copy_trace(json_data):
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


def test_metadata_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    # patch summary groups for testing purposes
    # (it appears sometimes these contain single quotes and other times it doesn't)
    data.metadata.config.summary_groups = [
        f"{itr}".strip("'") for itr in data.metadata.config.summary_groups
    ]

    num_summary_grps = len(data.metadata.config.summary_groups)
    expected_summary_grps = (
        [
            "KERNEL_DISPATCH|MEMORY_COPY",
        ]
        if num_summary_grps == 1
        else [
            "KERNEL_DISPATCH|MEMORY_COPY",
            ".*_API",
        ]
    )

    assert data.metadata.config.summary is True
    assert data.metadata.config.summary_per_domain is True
    assert data.metadata.config.summary_unit == "nsec"
    assert data.metadata.config.summary_file == "summary"
    assert data.metadata.config.summary_groups == expected_summary_grps

    assert len(data.metadata.command) == 4

    # patch command to make it easier to test
    data.metadata.command[0] = os.path.basename(data.metadata.command[0])

    assert data.metadata.command == ["transpose", "2", "500", "10"]


def test_summary_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    domains = []
    for itr in data.summary:
        domains.append(itr.domain)
        if itr.domain == "KERNEL_DISPATCH":
            assert itr.stats.count == 1004
            expected = dict([["run/iteration", 1000]])
            for oitr in itr.stats.operations:
                if oitr.key in expected.keys():
                    assert oitr.value.count == expected[oitr.key]
                else:
                    assert oitr.key.startswith("run/rank-")
                    assert (
                        re.match(
                            r"run/rank-([0-9]+)/thread-([0-9]+)/device-([0-9]+)/begin",
                            oitr.key,
                        )
                        is not None
                    )
                    assert oitr.value.count == 2
        elif itr.domain == "HIP_API":
            assert itr.stats.count >= 2130 and itr.stats.count <= 2165
        elif itr.domain == "MEMORY_COPY":
            # two threads + two memory copies (H2D + D2H).
            # HIP may decompose memory copies into more than one HSA memory copy
            assert itr.stats.count >= 4 and (itr.stats.count % 4) == 0
        elif itr.domain == "MEMORY_ALLOCATION":
            memory_allocation_allocate_count = 0
            memory_allocation_free_count = 0
            memory_allocation_vmem_allocate_count = 0
            memory_allocation_vmem_free_count = 0
            for operation in itr.stats.operations:
                if operation.key == "MEMORY_ALLOCATION_ALLOCATE":
                    memory_allocation_allocate_count = operation.value.count
                elif operation.key == "MEMORY_ALLOCATION_FREE":
                    memory_allocation_free_count = operation.value.count
                elif operation.key == "MEMORY_ALLOCATION_VMEM_ALLOCATE":
                    memory_allocation_vmem_allocate_count = operation.value.count
                elif operation.key == "MEMORY_ALLOCATION_VMEM_FREE":
                    memory_allocation_vmem_free_count = operation.value.count
            memory_allocation_allocate_and_free_count = (
                memory_allocation_allocate_count + memory_allocation_free_count
            )
            assert (
                memory_allocation_allocate_and_free_count >= 10
                and memory_allocation_allocate_and_free_count <= 30
            )
            # check if hip-runtime memory management pools through virtual memory allocation count is equal to free count.
            assert (
                memory_allocation_vmem_allocate_count == memory_allocation_vmem_free_count
            )
        elif itr.domain == "MARKER_API":
            assert itr.stats.count == 1106
            expected = dict(
                [
                    ["run", 2],
                    ["run/iteration", 1000],
                    ["run/iteration/sync", 100],
                ]
            )
            for oitr in itr.stats.operations:
                if oitr.key in expected.keys():
                    assert oitr.value.count == expected[oitr.key]
                else:
                    assert oitr.key.startswith("run/rank-")
                    assert (
                        re.match(
                            r"run/rank-([0-9]+)/thread-([0-9]+)/device-([0-9]+)/(begin|end)",
                            oitr.key,
                        )
                        is not None
                    )
                    assert oitr.value.count == 1
        else:
            assert False, f"unhandled domain: {itr.domain}"

    assert len(list(set(domains))) == len(domains)


def test_summary_display_data(json_data, summary_data):
    data = json_data["rocprofiler-sdk-tool"]
    num_summary_grps = len(data.metadata.config.summary_groups)

    def get_df(domain):
        return summary_data[domain]

    def get_dims(df):
        # return rows x cols
        return [df.shape[0], df.shape[1]] if df is not None else [0, 0]

    hip = get_df("HIP_API")
    marker = get_df("MARKER_API")
    dispatch = get_df("KERNEL_DISPATCH")
    memcpy = get_df("MEMORY_COPY")
    memalloc = get_df("MEMORY_ALLOCATION")
    dispatch_and_copy = get_df("KERNEL_DISPATCH + MEMORY_COPY")
    hip_and_marker = get_df("HIP_API + MARKER_API") if num_summary_grps > 1 else None
    total = get_df("SUMMARY")

    expected_hip_and_marker_dims = [21, 9] if hip_and_marker is not None else [0, 0]

    assert get_dims(marker) == [7, 9], f"{marker}"
    assert get_dims(memcpy) == [2, 9], f"{memcpy}"
    assert get_dims(memalloc) in (
        [2, 9],  # [2,9] when hip-runtime doesn't use vmem.
        [4, 9],
    ), f"{memalloc}"
    assert get_dims(dispatch) == [3, 9], f"{dispatch}"
    assert get_dims(dispatch_and_copy) == [5, 9], f"{dispatch_and_copy}"
    assert get_dims(hip) == [14, 9], f"{hip}"
    assert get_dims(hip_and_marker) == expected_hip_and_marker_dims, f"{hip_and_marker}"
    if get_dims(memalloc) == [2, 9]:  # [2,9] when hip-runtime doesn't use vmem alloc.
        assert get_dims(total) == [25, 9], f"{total}"
    elif get_dims(memalloc) == [4, 9]:
        assert get_dims(total) == [27, 9], f"{total}"


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy"),
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy", "memory_allocation"),
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
