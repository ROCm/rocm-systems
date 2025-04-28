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

counters_list = ["SQ_WAVES", "GRBM_GUI_ACTIVE"]


def test_validate_counter_collection_pmc1(
    input_data: pd.DataFrame, retain_agent_prefix_flag
):
    df = input_data
    assert not df.empty
    if retain_agent_prefix_flag == "true":
        df_agent_id = df["Agent_Id"].str.split(" ").str[-1]
        assert (df_agent_id.astype(int).values >= 0).all()
    else:
        assert (df["Agent_Id"].astype(int).values > 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert len(df["Kernel_Name"]) > 0

    for counter in counters_list:
        assert counter in df.columns.tolist()

    for counter in counters_list:
        for itr in df[counter].values:
            assert itr > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
