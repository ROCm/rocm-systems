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

from rocprofiler_sdk.pytest_utils.hardware_counters import (
    skip_if_hw_counter_values_unavailable,
)
from collections import defaultdict


def unique(lst):
    return list(set(lst))


def validate_csv(df, kernel_list, counter_name):

    assert not df.empty
    df_Agent_id = df["Agent_Id"].str.split(" ").str[-1]
    assert (df_Agent_id.astype(int).values >= 0).all()
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
    assert df["Counter_Name"].str.contains(counter_name).all()
    skip_if_hw_counter_values_unavailable(df["Counter_Value"])
    assert (df["Counter_Value"].astype(int).values > 0).all()


def validate_csv_per_agent_iteration_count(df, kernel_list, iteration_range):

    # range is applied per kernel per device: each (kernel, agent) pair must have
    # exactly one row per requested iteration (device-count independent)
    if iteration_range is None:
        pytest.skip("input config has no kernel_iteration_range to validate")

    expected_count = len(iteration_range)
    assert expected_count > 0

    filtered = df[~df["Kernel_Name"].str.contains(r"__amd_rocclr_.*", regex=True)]
    for kernel_name in kernel_list:
        per_kernel = filtered[filtered["Kernel_Name"] == kernel_name]
        assert not per_kernel.empty, f"no rows captured for {kernel_name}"
        for agent_id, per_agent in per_kernel.groupby("Agent_Id"):
            assert len(per_agent) == expected_count, (
                f"{kernel_name} on {agent_id} captured {len(per_agent)} "
                f"dispatches, expected {expected_count} (iteration range "
                f"{sorted(iteration_range)})"
            )


def validate_json_iteration_range(json_data, kernel_list, iteration_range):

    # kernel_dispatch is the unfiltered launch trace, counter_collection the
    # selected dispatches; assert the selected per-kernel ordinals equal the range
    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    # the range is applied per kernel per device, so ordinals are counted within
    # each (kernel, agent) pair rather than across the whole run
    targeted_dispatches = []
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        if kernel_name in kernel_list:
            targeted_dispatches.append(
                (
                    dispatch_info["dispatch_id"],
                    kernel_name,
                    dispatch_info["agent_id"]["handle"],
                )
            )

    # assign each dispatch its 1-based ordinal in dispatch_id order
    launches_seen = defaultdict(int)
    ordinal_by_dispatch_id = {}
    key_by_dispatch_id = {}
    for dispatch_id, kernel_name, agent_handle in sorted(targeted_dispatches):
        key = (kernel_name, agent_handle)
        launches_seen[key] += 1
        ordinal_by_dispatch_id[dispatch_id] = launches_seen[key]
        key_by_dispatch_id[dispatch_id] = key

    assert launches_seen, f"no launches recorded for {kernel_list}"

    # need more launches than the range so ordinals are recoverable
    expected_ordinals = set(iteration_range)
    for key, launches in launches_seen.items():
        assert launches > len(expected_ordinals), (
            f"{key[0]} on agent {key[1]} launched {launches} times; expected more "
            f"than the {len(expected_ordinals)} requested iterations so original "
            "launch ordinals can be recovered"
        )

    # list, not set, so duplicate records are visible
    captured_ordinals = defaultdict(list)
    for counter in counter_collection_data:
        dispatch_id = counter["dispatch_data"]["dispatch_info"]["dispatch_id"]
        key = key_by_dispatch_id.get(dispatch_id)
        if key is None:
            continue
        captured_ordinals[key].append(ordinal_by_dispatch_id[dispatch_id])

    assert set(captured_ordinals) == set(launches_seen), (
        f"captured kernel/agent pairs {sorted(captured_ordinals)} do not match "
        f"the launched pairs {sorted(launches_seen)}"
    )

    for key, ordinals in captured_ordinals.items():
        assert set(ordinals) == expected_ordinals, (
            f"{key[0]} on agent {key[1]} captured iterations {sorted(set(ordinals))}, "
            f"expected {sorted(expected_ordinals)}"
        )
        assert len(ordinals) == len(expected_ordinals), (
            f"{key[0]} on agent {key[1]} captured {len(ordinals)} records for "
            f"{len(expected_ordinals)} requested iterations {sorted(expected_ordinals)}"
        )


def validate_json(json_data, counter_name, check_dispatch):

    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    dispatch_ids = []
    # at present, AQLProfile has bugs when reporting the counters for below architectures
    skip_gfx = ("gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1152", "gfx1153")

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
            values = []
            for record in counter["records"]:
                counter = get_counter(record["counter_id"])
                assert counter is not None, f"record:\n\t{record}"
                assert (
                    counter["name"] == counter_name
                ), f"record:\n\t{record}\ncounter:\n\t{counter}"
                if agent["name"] not in skip_gfx:
                    values.append(record["value"])

            # Check aggregate sum
            if agent["name"] not in skip_gfx:
                assert sum(values) > 0, f"{counter_name} value is not > 0"

    if check_dispatch:
        di_uniq = list(set(sorted(dispatch_ids)))
        # make sure the dispatch ids are unique and ordered
        di_expect = [idx + 1 for idx in range(len(dispatch_ids))]
        assert di_expect == di_uniq


def test_validate_counter_collection_csv_pass1(
    input_csv_pass1: pd.DataFrame, iteration_range_pass1
):
    kernel_list = sorted(["addition_kernel", "subtract_kernel", "divide_kernel"])
    validate_csv(input_csv_pass1, kernel_list, "SQ_WAVES")
    validate_csv_per_agent_iteration_count(
        input_csv_pass1, kernel_list, iteration_range_pass1
    )


def test_validate_counter_collection_csv_pmc1(input_csv_pmc1: pd.DataFrame):
    kernel_list = sorted(["addition_kernel", "subtract_kernel", "divide_kernel"])
    validate_csv(input_csv_pmc1, kernel_list, "SQ_WAVES")


def test_validate_counter_collection_csv_iteration_range(
    input_csv_iteration_range: pd.DataFrame, iteration_range
):
    kernel_list = sorted(
        ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
    )
    validate_csv(input_csv_iteration_range, kernel_list, "SQ_WAVES")
    validate_csv_per_agent_iteration_count(
        input_csv_iteration_range, kernel_list, iteration_range
    )


def test_validate_counter_collection_json_iteration_range(
    input_json_iteration_range, iteration_range
):
    kernel_list = sorted(
        ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
    )
    validate_json_iteration_range(
        input_json_iteration_range, kernel_list, iteration_range
    )


def test_validate_counter_collection_csv_pass2(
    input_csv_pass2: pd.DataFrame, iteration_range_pass2
):
    kernel_list = sorted(
        ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
    )
    validate_csv(input_csv_pass2, kernel_list, "GRBM_COUNT")
    validate_csv_per_agent_iteration_count(
        input_csv_pass2, kernel_list, iteration_range_pass2
    )


def test_validate_counter_collection_csv_pass3(input_csv_pass3: pd.DataFrame):
    kernel_list = sorted(
        ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
    )
    validate_csv(input_csv_pass3, kernel_list, "GRBM_GUI_ACTIVE")


def test_validate_counter_collection_csv_pass4(input_csv_pass4: pd.DataFrame):
    kernel_list = sorted(["divide_kernel"])
    validate_csv(input_csv_pass4, kernel_list, "SQ_WAVES")


def test_validate_counter_collection_json_pass1(input_json_pass1):
    validate_json(input_json_pass1, "SQ_WAVES", False)


def test_validate_counter_collection_json_pass2(input_json_pass2):
    validate_json(input_json_pass2, "GRBM_COUNT", False)


def test_validate_counter_collection_json_pass3(input_json_pass3):
    validate_json(input_json_pass3, "GRBM_GUI_ACTIVE", True)


def test_validate_counter_collection_json_pass4(input_json_pass4):
    validate_json(input_json_pass4, "SQ_WAVES", False)


# --------------------------------------------------------------------------- #
# CLI filter coverage (kernel-include / kernel-exclude / kernel-iteration-range
# passed directly on the command line, plus negative / edge-case behavior)
# --------------------------------------------------------------------------- #


def _non_rocclr(df):
    return df[~df["Kernel_Name"].astype(str).str.contains(r"__amd_rocclr_.*", regex=True)]


def validate_csv_present_absent(df, present_kernels, absent_literals, counter_name):
    assert not df.empty
    filtered = _non_rocclr(df)
    names = sorted(filtered["Kernel_Name"].unique().tolist())
    assert names == sorted(present_kernels), f"unexpected kernel set: {names}"
    for absent in absent_literals:
        assert (
            not df["Kernel_Name"].astype(str).str.contains(absent, regex=False).any()
        ), f"'{absent}' should not appear anywhere in the output"
    assert filtered["Counter_Name"].str.contains(counter_name).all()
    assert (filtered["Counter_Value"].astype(int).values > 0).all()


def test_validate_cli_single_kernel_csv(input_csv_file, iteration_range):
    # --kernel-include-regex narrows to one kernel; the other three must not
    # appear anywhere, and the range must limit each (kernel, agent) pair.
    kernel_list = ["addition_kernel"]
    absent = ["subtract_kernel", "multiply_kernel", "divide_kernel"]
    validate_csv_present_absent(input_csv_file, kernel_list, absent, "SQ_WAVES")
    validate_csv_per_agent_iteration_count(input_csv_file, kernel_list, iteration_range)


def test_validate_cli_single_kernel_json(input_json_file, iteration_range):
    # exact-iteration (both-direction) proof for the single included kernel
    validate_json_iteration_range(input_json_file, ["addition_kernel"], iteration_range)


def test_validate_cli_exclude_csv(input_csv_file):
    # include everything, exclude one kernel: the excluded name must be gone
    kernel_list = ["addition_kernel", "subtract_kernel", "divide_kernel"]
    validate_csv_present_absent(
        input_csv_file, kernel_list, ["multiply_kernel"], "SQ_WAVES"
    )


def test_validate_cli_mangled_csv(input_csv_file):
    # regex matches the mangled symbol as a substring; names stay mangled
    filtered = _non_rocclr(input_csv_file)
    names = set(filtered["Kernel_Name"].astype(str).unique())
    assert len(names) >= 1, "expected the mangled addition kernel to be profiled"
    for name in names:
        assert "addition_kernel" in name
        assert re.match(r"_Z[0-9]+addition_kernel", name), f"expected mangled: {name}"
        assert name != "addition_kernel", "expected mangled, not truncated, name"
    for other in ("subtract", "multiply", "divide"):
        assert (
            not input_csv_file["Kernel_Name"]
            .astype(str)
            .str.contains(other, regex=False)
            .any()
        ), f"'{other}' kernel should not be profiled"


def test_validate_cli_empty_result_json(input_json_file):
    # zero counter records but no crash; kernel_dispatch proves the app ran
    data = input_json_file["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    assert (
        len(counter_collection_data) == 0
    ), f"expected zero counter_collection records, got {len(counter_collection_data)}"
    assert len(data["buffer_records"]["kernel_dispatch"]) > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
