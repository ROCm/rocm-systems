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

target_kernel_regex = re.compile(r"target_kernel|pc_sampling_kernel")
paused_kernel_regex = re.compile(r"kernel_add|kernel_mult")

# The test binary launches target_kernel and pc_sampling_kernel 4 times each
# inside resume/pause blocks (at iterations 0, 16, 32, 48 of a 64-iteration loop).
EXPECTED_TARGET_KERNELS = 4
EXPECTED_PC_SAMPLING_KERNELS = 4
EXPECTED_DISPATCH_COUNT = EXPECTED_TARGET_KERNELS + EXPECTED_PC_SAMPLING_KERNELS


def test_kernel_data(json_data):
    def get_kernel_name(kernel_id: int) -> str:
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    kernel_dispatch_data = buffer_records["kernel_dispatch"]
    assert len(kernel_dispatch_data) > 0

    # check buffering data
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        assert dispatch_info["kernel_id"] > 0

        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        assert (
            target_kernel_regex.search(kernel_name) is not None
        ), f"kernel '{kernel_name}' does not match regular expression '{target_kernel_regex.pattern}'"


def test_paused_kernels_excluded(json_data):
    """Verify that kernels launched while profiling is paused do not appear in the trace."""

    def get_kernel_name(kernel_id: int) -> str:
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    data = json_data["rocprofiler-sdk-tool"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]

    for dispatch in kernel_dispatch_data:
        kernel_name = get_kernel_name(dispatch["dispatch_info"]["kernel_id"])
        assert paused_kernel_regex.search(kernel_name) is None, (
            f"kernel '{kernel_name}' should NOT appear in trace — "
            f"it was launched while profiling was paused"
        )


def test_expected_dispatch_count(json_data):
    """Verify the expected number of kernel dispatches were recorded."""
    data = json_data["rocprofiler-sdk-tool"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]

    assert len(kernel_dispatch_data) == EXPECTED_DISPATCH_COUNT, (
        f"Expected {EXPECTED_DISPATCH_COUNT} kernel dispatches "
        f"({EXPECTED_TARGET_KERNELS} target + {EXPECTED_PC_SAMPLING_KERNELS} pc_sampling), "
        f"got {len(kernel_dispatch_data)}"
    )


def test_dispatch_timestamps_ordered(json_data):
    """Verify that kernel dispatch end timestamps are non-decreasing."""
    data = json_data["rocprofiler-sdk-tool"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]

    end_timestamps = []
    for dispatch in kernel_dispatch_data:
        end_ts = dispatch["end_timestamp"]
        assert end_ts > 0, "Dispatch end timestamp should be positive"
        end_timestamps.append(end_ts)

    for i in range(1, len(end_timestamps)):
        assert end_timestamps[i] >= end_timestamps[i - 1], (
            f"Dispatch end timestamps not ordered: "
            f"dispatch[{i - 1}]={end_timestamps[i - 1]} > dispatch[{i}]={end_timestamps[i]}"
        )


def test_counter_collection_data(json_data):
    def get_kernel_name(kernel_id: int) -> str:
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    assert len(counter_collection_data) > 0

    for counter in counter_collection_data:
        dispatch_data = counter["dispatch_data"]["dispatch_info"]

        assert dispatch_data["dispatch_id"] > 0
        assert dispatch_data["agent_id"]["handle"] > 0
        assert dispatch_data["queue_id"]["handle"] > 0

        kernel_name = get_kernel_name(dispatch_data["kernel_id"])
        assert (
            target_kernel_regex.search(kernel_name) is not None
        ), f"kernel '{kernel_name}' does not match regular expression '{target_kernel_regex.pattern}'"

    # Verify counter values are present and non-negative
    for counter in counter_collection_data:
        records = counter.get("records", [])
        assert len(records) > 0, "Counter collection record should have counter values"
        for record in records:
            assert record["value"] >= 0.0, (
                f"Counter value should be non-negative, got {record['value']}"
            )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
