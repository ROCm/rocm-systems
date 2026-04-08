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


def test_kernel_records(input_data):
    """verify all kernels are present in the output"""
    data = input_data

    expected_kernel_count = 20000
    cursor = data.cursor()

    cursor.execute("SELECT COUNT(*) FROM kernels")
    kernel_count = cursor.fetchone()[0]
    assert (
        kernel_count == expected_kernel_count
    ), f"Expected {expected_kernel_count} kernel records, but found {kernel_count}"


def test_counter_records(input_data):
    """verify GRBM_COUNT PMC events are an exact multiple of the number of kernels"""
    data = input_data

    cursor = data.cursor()

    kernel_count = cursor.execute("SELECT COUNT(*) FROM kernels").fetchone()[0]
    pmc_event_count = cursor.execute("SELECT COUNT(*) FROM rocpd_pmc_event").fetchone()[0]

    assert (pmc_event_count % kernel_count) == 0, (
        f"Expected rocpd_pmc_event count ({pmc_event_count}) to be exact multiple of "
        f"kernel count ({kernel_count})"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
