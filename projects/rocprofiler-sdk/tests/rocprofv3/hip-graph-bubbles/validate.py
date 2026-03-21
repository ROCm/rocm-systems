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


def test_kernel_trace_row_count(
    kernel_input_data,
    expected_dispatch_count,
    expected_kernels,
    expected_iterations,
):
    assert expected_dispatch_count == expected_kernels * expected_iterations
    assert len(kernel_input_data) == expected_dispatch_count


def test_kernel_trace_dispatch_ids(kernel_input_data, expected_dispatch_count):
    dispatch_ids = [int(row["Dispatch_Id"]) for row in kernel_input_data]

    assert len(dispatch_ids) == expected_dispatch_count
    assert len(set(dispatch_ids)) == expected_dispatch_count


def test_kernel_trace_fields(kernel_input_data, expected_dispatch_count):
    assert len(kernel_input_data) == expected_dispatch_count

    for row in kernel_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"].split(" ")[-1]) >= 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert int(row["Correlation_Id"]) > 0
        assert int(row["Workgroup_Size_X"]) == 256
        assert int(row["Workgroup_Size_Y"]) == 1
        assert int(row["Workgroup_Size_Z"]) == 1
        assert int(row["Grid_Size_X"]) == 256
        assert int(row["Grid_Size_Y"]) == 1
        assert int(row["Grid_Size_Z"]) == 1
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
