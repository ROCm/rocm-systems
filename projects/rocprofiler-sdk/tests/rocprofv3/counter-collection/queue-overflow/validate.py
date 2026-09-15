#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Regression coverage for the hsa-runtime InterceptQueue retry/overflow hang
# during counter collection. The multistream workload launches 100 streams x
# 100 iterations = 10000 kernel dispatches. Counter collection rewrites every
# dispatch (counter start/stop packets) via the hsa-runtime InterceptQueue, so
# a regression either hangs (the execute test hits its TIMEOUT) or drops
# dispatches / corrupts the ring copy. The assertions below require that all
# 10000 dispatches were captured with valid SQ_WAVES data.

import sys
import pytest
import pandas as pd
import re

# multistream: 100 streams x 100 iterations (see tests/bin/multistream/multistream_app.cpp)
EXPECTED_DISPATCHES = 100 * 100
KERNEL_NAME = "add(int, float*, float*)"

# AQLProfile has known counter mis-reporting bugs on these architectures, so the
# per-dispatch SQ_WAVES *value* is unreliable under the InterceptQueue overflow
# path (the structural checks remain valid). Shared by the CSV and JSON validators.
SKIP_GFX = ("gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1153")


def is_helper(name):
    return bool(re.search(r"__amd_rocclr_.*", name))


def dispatch_agent_names(json_data):
    # Names (e.g. "gfx1150") of the agents that actually ran the collected
    # dispatches, resolved from the JSON tool output.
    data = json_data["rocprofiler-sdk-tool"]
    agents_by_handle = {a["id"]["handle"]: a for a in data["agents"]}
    names = set()
    for counter in data["callback_records"]["counter_collection"]:
        info = counter["dispatch_data"]["dispatch_info"]
        agent = agents_by_handle.get(info["agent_id"]["handle"])
        if agent is not None:
            names.add(agent["name"])
    return names


def test_validate_counter_collection_queue_overflow(input_data: pd.DataFrame, json_data):
    df = input_data

    assert not df.empty
    df_agent_id = df["Agent_Id"].str.split(" ").str[-1]
    assert (df_agent_id.astype(int).values >= 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert (df["Process_Id"].astype(int).values > 0).all()

    # only real (non-rocclr-helper) kernel dispatched by multistream is the add kernel
    real_kernels = sorted(
        x for x in df["Kernel_Name"].unique().tolist() if not is_helper(x)
    )
    assert real_kernels == [KERNEL_NAME]

    # all counters requested are SQ_WAVES
    assert df["Counter_Name"].str.contains("SQ_WAVES").all()

    # Skip ONLY the per-value >0 check on architectures with known AQLProfile
    # counter mis-reporting under the InterceptQueue overflow path (mirrors the
    # JSON validator below). Every structural check above/below still runs.
    if not (dispatch_agent_names(json_data) & set(SKIP_GFX)):
        assert (df["Counter_Value"].astype(int).values > 0).all()

    # key regression check: every one of the 10000 dispatches was captured (a hang or
    # out-of-bounds ring copy in the overflow/retry path would strand or drop dispatches)
    add_rows = df[df["Kernel_Name"] == KERNEL_NAME]
    assert len(add_rows) == EXPECTED_DISPATCHES

    # dispatch ids are unique and contiguous from 1
    di_list = df["Dispatch_Id"].astype(int).values.tolist()
    di_uniq = sorted(df["Dispatch_Id"].unique().tolist())
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_uniq


def test_validate_counter_collection_queue_overflow_json(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]

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

    dispatch_ids = []
    add_dispatches = 0
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

        if is_helper(kernel_name):
            continue
        add_dispatches += 1

        sq_waves_values = []
        for record in counter["records"]:
            counter_meta = get_counter(record["counter_id"])
            assert counter_meta is not None, f"record:\n\t{record}"
            assert (
                counter_meta["name"] == "SQ_WAVES"
            ), f"record:\n\t{record}\ncounter:\n\t{counter_meta}"
            if agent["name"] not in SKIP_GFX:
                sq_waves_values.append(record["value"])

        if agent["name"] not in SKIP_GFX:
            assert sum(sq_waves_values) > 0, "SQ_WAVES value is not > 0"

    # key regression check: all 10000 dispatches captured, none stranded/dropped
    assert add_dispatches == EXPECTED_DISPATCHES

    di_uniq = sorted(set(dispatch_ids))
    di_expect = [idx + 1 for idx in range(len(dispatch_ids))]
    assert di_expect == di_uniq


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
