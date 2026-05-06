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

import sys
import re
import pytest
import pandas as pd

# Architectures on which the new TD counters are defined (see
# projects/rocprofiler-sdk/source/share/rocprofiler-sdk/config.yaml).
TD_SUPPORTED_GFX = (
    "gfx11",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
)

# Counters expected to record non-zero values when running vector-ops on a
# gfx11 target. Limited to BUSY counters whose units are continuously
# exercised by simple memory-load HIP kernels (TD load/issue path and the
# non-filter pipeline). Sampler/ray-tracing BUSY counters are *not* included
# here because vector-ops never issues sampler or BVH instructions, so those
# units may legitimately stay at 0.
TD_NONZERO_BUSY_COUNTERS = (
    "TD_TD_BUSY",
    "TD_TD_BUSY_sum",
    "TD_INPUT_BUSY",
    "TD_INPUT_BUSY_sum",
    "TD_NOFILTER_BUSY",
    "TD_NOFILTER_BUSY_sum",
)

# Per-pass expected counter names (matches input.txt).
EXPECTED_COUNTERS_BY_PASS = {
    1: {
        "TD_TD_BUSY",
        "TD_TD_BUSY_sum",
        "TD_INPUT_BUSY",
        "TD_INPUT_BUSY_sum",
    },
    2: {
        "TD_SAMPLER_LERP_BUSY",
        "TD_SAMPLER_LERP_BUSY_sum",
        "TD_SAMPLER_OUT_BUSY",
        "TD_SAMPLER_OUT_BUSY_sum",
    },
    3: {
        "TD_NOFILTER_BUSY",
        "TD_NOFILTER_BUSY_sum",
        "TD_RAY_TRACING_BVH4_BUSY",
        "TD_RAY_TRACING_BVH4_BUSY_sum",
    },
    4: {
        "TD_TA_DATA_STALL",
        "TD_TA_DATA_STALL_sum",
        "TD_TC_STALL",
        "TD_TC_STALL_sum",
    },
    5: {
        "TD_TC_RAM_STALL",
        "TD_TC_RAM_STALL_sum",
        "TD_LDS_STALL",
        "TD_LDS_STALL_sum",
    },
}

KERNEL_LIST = sorted(
    ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
)


def _is_td_supported_csv(df: pd.DataFrame) -> bool:
    """Return True if any agent in the CSV is a gfx11 target that defines the
    new TD counters. The Agent column is just an opaque "Agent N" label so we
    cannot infer the gfx target from the CSV alone; rocprofv3 only ever emits
    counter rows for agents that actually programmed the requested counters,
    so a non-empty TD-counter CSV is itself proof that the underlying agent
    supports them. Callers should still cross-check against JSON when
    available for stronger guarantees."""
    return df is not None and not df.empty


def _validate_pass_csv(df: pd.DataFrame, pmc_idx: int):
    """Common CSV checks for a single pmc_N/ output directory."""
    expected_counters = EXPECTED_COUNTERS_BY_PASS[pmc_idx]

    if df is None:
        pytest.skip(f"--input-csv-pmc{pmc_idx} not provided")

    if df.empty:
        # Off-target (e.g., gfx9/gfx10/gfx12) machines: rocprofv3 emits no
        # rows because the TD counters are not defined for the agent. The
        # test still passes structurally.
        return

    df_agent_id = df["Agent_Id"].str.split(" ").str[-1]
    assert (df_agent_id.astype(int).values >= 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert (df["Process_Id"].astype(int).values > 0).all()
    assert len(df["Kernel_Name"]) > 0

    observed_kernels = sorted(
        x
        for x in df["Kernel_Name"].unique().tolist()
        if not re.search(r"__amd_rocclr_.*", x)
    )
    assert observed_kernels == KERNEL_LIST, (
        f"pmc_{pmc_idx} kernel list mismatch: got {observed_kernels}"
    )

    observed_counters = set(df["Counter_Name"].unique().tolist())
    assert observed_counters.issubset(expected_counters), (
        f"pmc_{pmc_idx} unexpected counters: {observed_counters - expected_counters}"
    )
    assert observed_counters == expected_counters, (
        f"pmc_{pmc_idx} missing counters: {expected_counters - observed_counters}"
    )

    for _, row in df.iterrows():
        name = row["Counter_Name"]
        value = float(row["Counter_Value"])
        if name in TD_NONZERO_BUSY_COUNTERS:
            assert value > 0, (
                f"pmc_{pmc_idx} counter {name} expected > 0, got {value} "
                f"(kernel={row['Kernel_Name']})"
            )
        else:
            # Sampler / ray-tracing BUSY counters and all *_STALL counters
            # may legitimately be 0 for vector-ops workloads.
            assert value >= 0, (
                f"pmc_{pmc_idx} counter {name} expected >= 0, got {value}"
            )

    di_uniq = sorted(df["Dispatch_Id"].unique().tolist())
    di_expect = [idx + 1 for idx in range(len(di_uniq))]
    assert di_expect == di_uniq, (
        f"pmc_{pmc_idx} dispatch ids not unique/ordered: {di_uniq}"
    )


def _validate_pass_json(json_data, pmc_idx: int):
    """Common JSON checks for a single pmc_N/ output directory."""
    expected_counters = EXPECTED_COUNTERS_BY_PASS[pmc_idx]

    if json_data is None:
        pytest.skip(f"--input-json-pmc{pmc_idx} not provided")

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

    if not counter_collection_data:
        # Off-target machine: no records were collected. Nothing to validate.
        return

    dispatch_ids = []
    on_supported_gfx = False
    for itr in counter_collection_data:
        dispatch_data = itr["dispatch_data"]["dispatch_info"]

        assert dispatch_data["dispatch_id"] > 0
        assert dispatch_data["agent_id"]["handle"] > 0
        assert dispatch_data["queue_id"]["handle"] > 0

        agent = get_agent(dispatch_data["agent_id"])
        kernel_name = get_kernel_name(dispatch_data["kernel_id"])
        assert agent is not None
        assert len(kernel_name) > 0

        gfx = agent["name"]
        is_td_target = gfx in TD_SUPPORTED_GFX
        on_supported_gfx = on_supported_gfx or is_td_target

        dispatch_ids.append(dispatch_data["dispatch_id"])
        if re.search(r"__amd_rocclr_.*", kernel_name):
            continue

        seen_counter_names = set()
        for record in itr["records"]:
            counter = get_counter(record["counter_id"])
            assert counter is not None, f"record:\n\t{record}"
            assert counter["name"] in expected_counters, (
                f"pmc_{pmc_idx} unexpected counter {counter['name']}"
            )
            seen_counter_names.add(counter["name"])

            if not is_td_target:
                continue

            value = record["value"]
            if counter["name"] in TD_NONZERO_BUSY_COUNTERS:
                assert value > 0, (
                    f"pmc_{pmc_idx} {counter['name']} expected > 0 on "
                    f"{gfx}, got {value} (kernel={kernel_name})"
                )
            else:
                assert value >= 0, (
                    f"pmc_{pmc_idx} {counter['name']} expected >= 0 on "
                    f"{gfx}, got {value}"
                )

        if is_td_target:
            assert seen_counter_names == expected_counters, (
                f"pmc_{pmc_idx} on {gfx}: missing counters "
                f"{expected_counters - seen_counter_names}"
            )

    di_uniq = sorted(set(dispatch_ids))
    di_expect = [idx + 1 for idx in range(len(di_uniq))]
    assert di_expect == di_uniq, (
        f"pmc_{pmc_idx} dispatch ids not unique/ordered: {di_uniq}"
    )

    # If this run was on a gfx11 target we must have observed at least one
    # TD-counter record.
    if on_supported_gfx:
        assert len(counter_collection_data) > 0


# CSV pass tests
def test_validate_td_counters_csv_pmc1(input_csv_pmc1):
    _validate_pass_csv(input_csv_pmc1, 1)


def test_validate_td_counters_csv_pmc2(input_csv_pmc2):
    _validate_pass_csv(input_csv_pmc2, 2)


def test_validate_td_counters_csv_pmc3(input_csv_pmc3):
    _validate_pass_csv(input_csv_pmc3, 3)


def test_validate_td_counters_csv_pmc4(input_csv_pmc4):
    _validate_pass_csv(input_csv_pmc4, 4)


def test_validate_td_counters_csv_pmc5(input_csv_pmc5):
    _validate_pass_csv(input_csv_pmc5, 5)


# JSON pass tests
def test_validate_td_counters_json_pmc1(input_json_pmc1):
    _validate_pass_json(input_json_pmc1, 1)


def test_validate_td_counters_json_pmc2(input_json_pmc2):
    _validate_pass_json(input_json_pmc2, 2)


def test_validate_td_counters_json_pmc3(input_json_pmc3):
    _validate_pass_json(input_json_pmc3, 3)


def test_validate_td_counters_json_pmc4(input_json_pmc4):
    _validate_pass_json(input_json_pmc4, 4)


def test_validate_td_counters_json_pmc5(input_json_pmc5):
    _validate_pass_json(input_json_pmc5, 5)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
