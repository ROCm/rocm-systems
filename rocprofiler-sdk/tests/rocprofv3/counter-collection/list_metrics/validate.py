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

import pandas as pd
import sys
import pytest


def test_validate_list_basic_metrics(basic_metrics_input_data):
    for row in basic_metrics_input_data:
        assert row["Agent_Id"].isdigit() == True
        assert row["Name"] != ""
        assert row["Description"] != ""
        assert row["Block"] != ""
        assert row["Dimensions"] != ""
        if row["Name"] == "SQ_WAVES":
            row[
                "Description"
            ] == "Count number of waves sent to SQs. (per-simd, emulated, global)"
            row["Block"] == "SQ"


def test_validate_list_derived_metrics(derived_metrics_input_data):
    for row in derived_metrics_input_data:
        assert row["Agent_Id"].isdigit() == True
        assert row["Name"] != ""
        assert row["Description"] != ""
        assert row["Expression"] != ""
        assert row["Dimensions"] != ""
        if row["Name"] == "TA_BUSY_min":
            row["Description"] == "TA block is busy. Min over TA instances."
            row["Expression"] == "reduce(TA_TA_BUSY,min)"


def test_validate_list_pc_sample_config(pc_sample_config_input_data):
    for row in pc_sample_config_input_data:
        assert row["Agent_Id"].isdigit() == True
        assert row["Method"] != ""
        assert row["Unit"] != ""
        assert row["Minimum_Interval"].isdigit() == True
        assert row["Maximum_Interval"].isdigit() == True


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
