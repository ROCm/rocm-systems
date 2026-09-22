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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
