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

kernel_list = sorted(
    ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
)


def unique(lst):
    return list(set(lst))


def test_validate_counter_collection_pmc1_extra_counters(input_data: pd.DataFrame):
    df = input_data

    assert not df.empty
    df_agent_id = df["Agent_Id"].str.split(" ").str[-1]
    assert (df_agent_id.astype(int).values >= 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert (df["Process_Id"].astype(int).values > 0).all()
    assert len(df["Kernel_Name"]) > 0

    counter_collection_pmc1_kernel_list = [
        x
        for x in sorted(df["Kernel_Name"].unique().tolist())
        if not re.search(r"__amd_rocclr_.*", x)
    ]

    assert kernel_list == counter_collection_pmc1_kernel_list

    kernel_count = dict([[itr, 0] for itr in kernel_list])
    assert len(kernel_count) == len(kernel_list)
    for itr in df["Kernel_Name"]:
        if re.search(r"__amd_rocclr_.*", itr):
            continue
        kernel_count[itr] += 1
    kn_cnt = [itr for _, itr in kernel_count.items()]
    assert min(kn_cnt) == max(kn_cnt) and len(unique(kn_cnt)) == 1

    assert len(df["Counter_Value"]) > 0
    assert df["Counter_Name"].str.contains("TEST_YAML_LOAD").all()
    assert (df["Counter_Value"].astype(int).values > 0).all()

    di_list = df["Dispatch_Id"].astype(int).values.tolist()
    di_uniq = sorted(df["Dispatch_Id"].unique().tolist())
    # make sure the dispatch ids are unique and ordered
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_uniq


def test_mixed_valid_invalid_counters(input_data: pd.DataFrame):
    """
    Verify that valid counters work even when the YAML file contains invalid ones.
    This tests graceful degradation - invalid entries should be skipped without
    breaking the collection of valid counters.
    """
    df = input_data

    # Basic sanity checks
    assert not df.empty
    assert len(df["Counter_Value"]) > 0

    # Get all counter names that were actually collected
    collected_counters = set(df["Counter_Name"].unique())

    # The valid counter should be present
    assert (
        "TEST_YAML_LOAD" in collected_counters
    ), f"Expected TEST_YAML_LOAD counter, got: {collected_counters}"

    # Invalid counter should NOT be present
    assert (
        "INVALID_COUNTER_NO_PAYLOAD" not in collected_counters
    ), "Invalid counter should not appear in output"

    # All collected counter values should be valid (non-negative)
    assert (df["Counter_Value"].astype(int).values >= 0).all()

    # Ensure all counter names are TEST_YAML_LOAD (invalid ones should be skipped)
    assert df["Counter_Name"].str.contains("TEST_YAML_LOAD").all()


def test_duplicate_counter_handling(input_data: pd.DataFrame):
    """
    Verify that duplicate counter definitions don't create corrupt output.
    The tool should handle duplicates gracefully - either using the first definition
    or producing a clean error, but not generating invalid data.
    """
    df = input_data

    # Basic sanity checks
    assert not df.empty

    # Get unique counter names
    counter_names = df["Counter_Name"].unique().tolist()

    # The duplicate counter should appear in output
    assert "TEST_YAML_LOAD" in counter_names

    # Verify all counter values are valid
    assert (df["Counter_Value"].astype(int).values >= 0).all()

    # Ensure all counter names are TEST_YAML_LOAD (duplicates should be resolved)
    assert df["Counter_Name"].str.contains("TEST_YAML_LOAD").all()

    # Check that duplicate doesn't cause malformed output
    # Each dispatch should have a consistent set of counters
    for dispatch_id in df["Dispatch_Id"].unique():
        dispatch_data = df[df["Dispatch_Id"] == dispatch_id]
        # All rows for this dispatch should have the same counter name
        # (no duplicate columns for the same counter)
        counter_counts = dispatch_data["Counter_Name"].value_counts()
        assert (
            counter_counts <= 1
        ).all(), f"Duplicate counter appears multiple times in dispatch {dispatch_id}: {counter_counts}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
