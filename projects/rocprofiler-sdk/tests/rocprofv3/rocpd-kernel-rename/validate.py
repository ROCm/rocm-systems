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


def test_csv_data(
    rename_csv_data,
    no_rename_csv_data,
    generated_rename_csv_data,
    generated_no_rename_csv_data,
):
    assert len(rename_csv_data) > 0, "Expected non-empty kernel rename csv data"
    assert (
        len(no_rename_csv_data) > 0
    ), "Expected non-empty non-kernel rename kernel csv data"
    assert (
        len(generated_rename_csv_data) > 0
    ), "Expected non-empty rocpd kernel rename csv data"
    assert (
        len(generated_no_rename_csv_data) > 0
    ), "Expected non-empty rocpd non-kernel rename csv data"

    for row, generated_row in zip(rename_csv_data, generated_rename_csv_data):
        assert row["Kernel_Name"] == generated_row["Kernel_Name"]

    for row, generated_row in zip(no_rename_csv_data, generated_no_rename_csv_data):
        assert row["Kernel_Name"] == generated_row["Kernel_Name"]


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
