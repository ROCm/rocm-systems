#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import Counter
import sys

import pytest

EXPECTED_KERNEL_COUNTS = {
    "target_kernel": 4,
    "pc_sampling_kernel": 4,
    "nested_kernel": 1,
}
EXCLUDED_KERNELS = ("kernel_add", "kernel_mult")
EXPECTED_RANGE_COUNTS = {
    "record_function: target": 4,
    "record_function: outer": 1,
    "record_function: inner": 1,
}


def get_kernel_names(data, records):
    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    return [get_kernel_name(record["dispatch_info"]["kernel_id"]) for record in records]


def check_only_record_function_kernels(kernel_names):
    assert kernel_names, "no kernels were collected inside record_function ranges"

    for excluded in EXCLUDED_KERNELS:
        leaked = [name for name in kernel_names if excluded in name]
        assert not leaked, (
            f"kernel '{excluded}' runs only outside record_function ranges but "
            f"produced record(s): {leaked}"
        )

    for kernel, expected_count in EXPECTED_KERNEL_COUNTS.items():
        actual_count = sum(kernel in name for name in kernel_names)
        assert actual_count == expected_count, (
            f"expected {expected_count} record(s) for '{kernel}' inside "
            f"record_function ranges but found {actual_count}"
        )

    assert len(kernel_names) == sum(EXPECTED_KERNEL_COUNTS.values())


def test_kernel_dispatch_only_inside_record_function_ranges(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    records = data["buffer_records"]["kernel_dispatch"]
    check_only_record_function_kernels(get_kernel_names(data, records))


def test_counter_collection_only_inside_record_function_ranges(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    records = [
        counter["dispatch_data"]
        for counter in data["callback_records"]["counter_collection"]
    ]
    check_only_record_function_kernels(get_kernel_names(data, records))


def test_record_function_ranges_are_traced(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    marker_names = {
        marker["key"]: marker["value"] for marker in data["strings"]["marker_api"]
    }
    range_names = []
    for marker in data["buffer_records"]["marker_api"]:
        kind = data["strings"]["buffer_records"][marker["kind"]]["kind"]
        if kind == "MARKER_CORE_RANGE_API":
            range_names.append(marker_names[marker["correlation_id"]["internal"]])

    counts = Counter(name for name in range_names if name.startswith("record_function:"))
    assert counts == Counter(EXPECTED_RANGE_COUNTS), (
        "record_function range collection did not match the active resume windows: "
        f"expected {Counter(EXPECTED_RANGE_COUNTS)}, found {counts}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
