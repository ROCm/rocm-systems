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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Validate --pmc SQ_WAVES --selected-regions: counters only inside roctx resume window."""

import re
import sys

import pytest

# Kernels that run inside a roctxProfilerResume / roctxProfilerPause window.
INCLUDED_KERNEL_REGEX = re.compile(r"target_kernel|pc_sampling_kernel")

# Kernels that are only ever dispatched while profiling is paused.
EXCLUDED_KERNELS = ("kernel_add", "kernel_mult", "nested_kernel")

# Each of the 4 resume windows launches target_kernel and pc_sampling_kernel once.
EXPECTED_KERNEL_COUNTS = {
    "target_kernel": 4,
    "pc_sampling_kernel": 4,
}


def test_counter_collection_only_inside_selected_regions(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    counter_collection_data = data["callback_records"]["counter_collection"]

    # (b) counter collection actually happened inside the resume window.
    assert len(counter_collection_data) > 0, "no counter records were collected"

    observed_kernel_names = []
    for counter in counter_collection_data:
        dispatch_data = counter["dispatch_data"]["dispatch_info"]

        assert dispatch_data["dispatch_id"] > 0
        assert dispatch_data["agent_id"]["handle"] > 0
        assert dispatch_data["queue_id"]["handle"] > 0

        observed_kernel_names.append(get_kernel_name(dispatch_data["kernel_id"]))

    # every counter record corresponds to a kernel inside the resume window.
    for kernel_name in observed_kernel_names:
        assert INCLUDED_KERNEL_REGEX.search(kernel_name) is not None, (
            f"counter record for kernel '{kernel_name}' is outside the resume window "
            f"(expected match for '{INCLUDED_KERNEL_REGEX.pattern}')"
        )

    # kernels dispatched while profiling was paused must be excluded entirely.
    for excluded in EXCLUDED_KERNELS:
        leaked = [name for name in observed_kernel_names if excluded in name]
        assert not leaked, (
            f"kernel '{excluded}' runs only while profiling is paused but produced "
            f"counter record(s): {leaked}"
        )

    # exactly the dispatches from the 4 resume windows were counted.
    for kernel, expected_count in EXPECTED_KERNEL_COUNTS.items():
        actual_count = sum(1 for name in observed_kernel_names if kernel in name)
        assert actual_count == expected_count, (
            f"expected {expected_count} counter record(s) for '{kernel}' "
            f"(one per resume window) but found {actual_count}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
