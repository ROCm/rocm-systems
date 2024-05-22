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


def test_validate_counter_collection_pmc1(input_data: pd.DataFrame):
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
    assert df["Counter_Name"].str.contains("SQ_WAVES").all()
    assert (df["Counter_Value"].astype(int).values > 0).all()

    di_list = df["Dispatch_Id"].astype(int).values.tolist()
    di_uniq = sorted(df["Dispatch_Id"].unique().tolist())
    # make sure the dispatch ids are unique and ordered
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_uniq


def test_validate_counter_collection_pmc1_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    dispatch_ids = []
    # at present, AQLProfile has bugs when reporting the counters for below architectures
    skip_gfx = ("gfx1101", "gfx1102")

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    def get_agent(agent_id):
        for agent in data["agents"]:
            if agent["id"]["handle"] == agent_id["handle"]:
                return agent
        return None

    def get_counter(counter_id):
        for counter in data["counters"]:
            if counter["id"]["handle"] == counter_id["handle"]:
                return counter
        return None

    for counter in counter_collection_data:
        dispatch_data = counter["dispatch_data"]["dispatch_info"]

        assert dispatch_data["dispatch_id"] > 0
        assert dispatch_data["agent_id"]["handle"] > 0
        assert dispatch_data["queue_id"]["handle"] > 0

        agent = get_agent(dispatch_data["agent_id"])
        kernel_name = get_kernel_name(dispatch_data["kernel_id"])

        assert agent is not None
        assert len(kernel_name) > 0

        dispatch_ids.append(dispatch_data["dispatch_id"])
        if not re.search(r"__amd_rocclr_.*", kernel_name):
            for record in counter["records"]:
                counter = get_counter(record["counter_id"])
                assert counter is not None, f"record:\n\t{record}"
                assert (
                    counter["name"] == "SQ_WAVES"
                ), f"record:\n\t{record}\ncounter:\n\t{counter}"
                if agent["name"] not in skip_gfx:
                    assert (
                        record["value"] > 0
                    ), f"record: {record}\ncounter: {counter}\nagent: {agent}"

    di_uniq = list(set(sorted(dispatch_ids)))
    # make sure the dispatch ids are unique and ordered
    di_expect = [idx + 1 for idx in range(len(dispatch_ids))]
    assert di_expect == di_uniq


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
