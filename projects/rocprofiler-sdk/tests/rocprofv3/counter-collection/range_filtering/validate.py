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
import numpy as np
import pandas as pd
import re

# the profiled kernel; internal rocclr helper kernels are ignored
TARGET_KERNEL = "transpose"


def unique(lst):
    return list(set(lst))


def validate_json(input_data):
    json_data = input_data["json_data"]
    iteration_list = input_data["iteration_range"]

    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    expected_iterations = set(iteration_list)
    assert len(expected_iterations) > 0, "iteration range must not be empty"

    # recover each launch's 1-based ordinal from the unfiltered kernel_dispatch
    # trace, ordered by dispatch_id
    launch_dispatch_ids = sorted(
        dispatch["dispatch_info"]["dispatch_id"]
        for dispatch in kernel_dispatch_data
        if get_kernel_name(dispatch["dispatch_info"]["kernel_id"]) == TARGET_KERNEL
    )
    ordinal_by_dispatch_id = {
        dispatch_id: ordinal + 1
        for ordinal, dispatch_id in enumerate(launch_dispatch_ids)
    }

    # need more launches than the range so the ordinals are recoverable
    assert len(launch_dispatch_ids) > len(expected_iterations), (
        f"{TARGET_KERNEL} launched {len(launch_dispatch_ids)} times; expected "
        f"more than the {len(expected_iterations)} requested iterations so the "
        "original launch ordinals can be recovered"
    )

    # map each counter-collected dispatch back to its ordinal (list keeps dups)
    captured_ordinals = []
    for counter in counter_collection_data:
        dispatch_info = counter["dispatch_data"]["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        if kernel_name != TARGET_KERNEL:
            continue
        dispatch_id = dispatch_info["dispatch_id"]
        assert dispatch_id in ordinal_by_dispatch_id, (
            f"counter record dispatch_id {dispatch_id} not found in the "
            f"{TARGET_KERNEL} kernel_dispatch trace"
        )
        captured_ordinals.append(ordinal_by_dispatch_id[dispatch_id])

    captured_set = set(captured_ordinals)

    # captured iterations must equal the requested set exactly
    assert captured_set == expected_iterations, (
        f"{TARGET_KERNEL} captured iterations {sorted(captured_set)}, "
        f"expected {sorted(expected_iterations)}"
    )

    # exactly one record per requested iteration (no duplicates, no misses)
    assert len(captured_ordinals) == len(expected_iterations), (
        f"{TARGET_KERNEL} captured {len(captured_ordinals)} records "
        f"(iterations {sorted(captured_ordinals)}); expected exactly "
        f"{len(expected_iterations)} ({sorted(expected_iterations)})"
    )


def test_validate_counter_collection_pass1(input_json_pass1):
    validate_json(input_json_pass1)


def test_validate_counter_collection_pass2(input_json_pass2):
    validate_json(input_json_pass2)


def test_validate_counter_collection_pass3(input_json_pass3):
    validate_json(input_json_pass3)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
