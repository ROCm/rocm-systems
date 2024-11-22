#!/usr/bin/env python3

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
    assert (df["Agent_Id"].astype(int).values > 0).all()
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
