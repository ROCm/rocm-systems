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
import pandas as pd
import re

# vector-ops kernels expected in every pass
KERNEL_LIST = sorted(
    ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
)

# SQG counters are defined for these gfx11 architectures.
SQG_SUPPORTED_GFX = (
    "gfx11",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
)

# At present, AQLProfile has bugs when reporting counters for these gfx11 SKUs
# (matches the skip list used by input1/validate.py). Plumbing/shape is still
# verified for these agents; only the strict "value > 0" checks are relaxed.
SKIP_GFX_VALUE_CHECK = (
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
)

# SQG counters that must aggregate to a non-zero value for any gfx11 compute
# workload that successfully launches a wave. SQG_WAVES_ENDED is excluded
# because on observed gfx11 hardware (gfx1151) the AQL sampling window
# captures wave starts but the END signal is not registered before the
# Stop+Read packets fire, so SQG_WAVES_ENDED can legitimately be 0.
ALWAYS_NONZERO = {
    "SQG_CYCLES",
    "SQG_BUSY_CYCLES",
    "SQG_WAVES",
    "SQG_WAVE_CYCLES",
    "SQG_ITEMS",
    "SQG_LEVEL_WAVES",
    "SQG_WAVES_STARTED",
}

# SQG counters that hardware guarantees to be zero ("perf counter is disabled").
MUST_BE_ZERO = {
    "SQG_NONE",
}

# Pairs where exactly one member must aggregate > 0 (depends on which wave size
# the compiler emitted for the kernel).
WAVE_SPLIT_PAIRS = [
    ("SQG_WAVES_32", "SQG_WAVES_64"),
    ("SQG_WAVE32_ITEMS", "SQG_WAVE64_ITEMS"),
    ("SQG_WAVES_EQ_32", "SQG_WAVES_EQ_64"),
]


def _unique(lst):
    return list(set(lst))


def _filter_kernels(names):
    return [x for x in names if not re.search(r"__amd_rocclr_.*", x)]


def _validate_csv_shape(df: pd.DataFrame, expected_kernels):
    """Generic CSV plumbing checks shared by every SQG pass."""

    assert not df.empty
    df_agent_id = df["Agent_Id"].str.split(" ").str[-1]
    assert (df_agent_id.astype(int).values >= 0).all()
    assert (df["Queue_Id"].astype(int).values > 0).all()
    assert (df["Process_Id"].astype(int).values > 0).all()
    assert len(df["Kernel_Name"]) > 0

    observed_kernels = sorted(_unique(_filter_kernels(df["Kernel_Name"].tolist())))
    assert (
        sorted(expected_kernels) == observed_kernels
    ), f"Expected kernels {sorted(expected_kernels)}, observed {observed_kernels}"

    kernel_count = {k: 0 for k in expected_kernels}
    for itr in df["Kernel_Name"]:
        if re.search(r"__amd_rocclr_.*", itr):
            continue
        kernel_count[itr] += 1
    counts = list(kernel_count.values())
    assert (
        min(counts) == max(counts) and len(_unique(counts)) == 1
    ), f"Kernel counts are not uniform: {kernel_count}"


def _validate_csv_counter_set(df: pd.DataFrame, expected_counters):
    """Assert every CSV row references one of ``expected_counters`` and that
    every expected counter is observed at least once."""

    expected_set = set(expected_counters)
    observed = set(df["Counter_Name"].unique())
    extras = observed - expected_set
    assert not extras, f"Unexpected counter names in CSV: {sorted(extras)}"
    missing = expected_set - observed
    assert not missing, f"SQG counters missing from CSV output: {sorted(missing)}"


def _validate_csv_values(df: pd.DataFrame, expected_counters):
    """Per-counter value validation:
    - every value is a non-negative integer,
    - counters in ``ALWAYS_NONZERO`` and the ``WAVE_SPLIT_PAIRS`` aggregate > 0,
    - counters in ``MUST_BE_ZERO`` aggregate exactly to 0.

    The strict > 0 checks are skipped when no agent in the CSV is a non-skip
    gfx11 SKU (mirrors the input1 behaviour). The == 0 check for MUST_BE_ZERO
    is unconditional because it reflects a hardware guarantee.
    """

    assert (
        df["Counter_Value"].astype(int).values >= 0
    ).all(), "Found negative SQG counter value in CSV output"

    for counter_name in MUST_BE_ZERO & set(expected_counters):
        sub = df[df["Counter_Name"] == counter_name]
        total = int(sub["Counter_Value"].astype(int).sum())
        assert total == 0, (
            f"{counter_name} is documented as 'do not count anything' but"
            f" aggregated to {total}"
        )

    for counter_name in ALWAYS_NONZERO & set(expected_counters):
        sub = df[df["Counter_Name"] == counter_name]
        assert not sub.empty, f"{counter_name} not present in CSV"
        total = int(sub["Counter_Value"].astype(int).sum())
        assert total > 0, f"{counter_name} aggregate is not > 0 (got {total})"

    for a, b in WAVE_SPLIT_PAIRS:
        if a in expected_counters and b in expected_counters:
            tot_a = int(df[df["Counter_Name"] == a]["Counter_Value"].astype(int).sum())
            tot_b = int(df[df["Counter_Name"] == b]["Counter_Value"].astype(int).sum())
            assert (
                tot_a + tot_b > 0
            ), f"Neither {a} nor {b} aggregated above 0 (got {tot_a} + {tot_b})"


def _validate_json_shape(json_data, expected_counters):
    """JSON plumbing + per-counter aggregate checks shared by every SQG pass."""

    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    dispatch_ids = []

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

    expected_set = set(expected_counters)

    counter_sums_active = {name: 0 for name in expected_counters}
    counter_seen_on_active_agent = {name: False for name in expected_counters}
    counter_must_be_zero_total = {name: 0 for name in MUST_BE_ZERO & expected_set}

    for entry in counter_collection_data:
        dispatch_data = entry["dispatch_data"]["dispatch_info"]

        assert dispatch_data["dispatch_id"] > 0
        assert dispatch_data["agent_id"]["handle"] > 0
        assert dispatch_data["queue_id"]["handle"] > 0

        agent = get_agent(dispatch_data["agent_id"])
        kernel_name = get_kernel_name(dispatch_data["kernel_id"])

        assert agent is not None
        assert len(kernel_name) > 0

        dispatch_ids.append(dispatch_data["dispatch_id"])

        if re.search(r"__amd_rocclr_.*", kernel_name):
            continue

        agent_skipped = agent["name"] in SKIP_GFX_VALUE_CHECK

        for record in entry["records"]:
            counter_obj = get_counter(record["counter_id"])
            assert counter_obj is not None, f"record:\n\t{record}"
            assert (
                counter_obj["name"] in expected_set
            ), f"unexpected counter: {counter_obj['name']}"

            if counter_obj["name"] in counter_must_be_zero_total:
                counter_must_be_zero_total[counter_obj["name"]] += record["value"]

            if not agent_skipped:
                counter_sums_active[counter_obj["name"]] += record["value"]
                counter_seen_on_active_agent[counter_obj["name"]] = True

    for name, total in counter_must_be_zero_total.items():
        assert (
            total == 0
        ), f"{name} is documented as 'do not count anything' but aggregated to {total}"

    for name in ALWAYS_NONZERO & expected_set:
        if counter_seen_on_active_agent[name]:
            assert (
                counter_sums_active[name] > 0
            ), f"{name} aggregate is not > 0 across non-skip agents"

    for a, b in WAVE_SPLIT_PAIRS:
        if (
            a in expected_set
            and b in expected_set
            and (counter_seen_on_active_agent[a] or counter_seen_on_active_agent[b])
        ):
            total = counter_sums_active[a] + counter_sums_active[b]
            assert total > 0, (
                f"Neither {a} nor {b} aggregated above 0 across non-skip agents"
                f" (got {counter_sums_active[a]} + {counter_sums_active[b]})"
            )

    di_uniq = sorted(set(dispatch_ids))
    di_expect = list(range(1, len(di_uniq) + 1))
    assert (
        di_expect == di_uniq
    ), f"dispatch ids are not unique/ordered: got {di_uniq}, expected {di_expect}"


# ----------------------------------------------------------------------
# pmc1: always-fire SQG counters (any gfx11 wave dispatch must produce > 0)
# ----------------------------------------------------------------------


PMC1_COUNTERS = [
    "SQG_CYCLES",
    "SQG_BUSY_CYCLES",
    "SQG_WAVES",
    "SQG_WAVE_CYCLES",
    "SQG_ITEMS",
    "SQG_LEVEL_WAVES",
    "SQG_WAVES_STARTED",
    "SQG_WAVES_ENDED",
]


def test_validate_sqg_counter_collection_csv_pmc1(input_csv_pmc1: pd.DataFrame):
    _validate_csv_shape(input_csv_pmc1, KERNEL_LIST)
    _validate_csv_counter_set(input_csv_pmc1, PMC1_COUNTERS)
    _validate_csv_values(input_csv_pmc1, PMC1_COUNTERS)


def test_validate_sqg_counter_collection_json_pmc1(input_json_pmc1):
    _validate_json_shape(input_json_pmc1, PMC1_COUNTERS)


# ----------------------------------------------------------------------
# pmc2: wave32/wave64 occupancy split + wave-fill bins
# ----------------------------------------------------------------------


PMC2_COUNTERS = [
    "SQG_WAVES_32",
    "SQG_WAVES_64",
    "SQG_WAVE32_ITEMS",
    "SQG_WAVE64_ITEMS",
    "SQG_WAVES_EQ_32",
    "SQG_WAVES_EQ_64",
    "SQG_WAVES_LT_64",
    "SQG_WAVES_LT_48",
]


def test_validate_sqg_counter_collection_csv_pmc2(input_csv_pmc2: pd.DataFrame):
    _validate_csv_shape(input_csv_pmc2, KERNEL_LIST)
    _validate_csv_counter_set(input_csv_pmc2, PMC2_COUNTERS)
    _validate_csv_values(input_csv_pmc2, PMC2_COUNTERS)


def test_validate_sqg_counter_collection_json_pmc2(input_json_pmc2):
    _validate_json_shape(input_json_pmc2, PMC2_COUNTERS)


# ----------------------------------------------------------------------
# pmc3: wave-fill bins, CWSR save/restore, internal counters, SQG_NONE
# ----------------------------------------------------------------------


PMC3_COUNTERS = [
    "SQG_WAVES_LT_32",
    "SQG_WAVES_LT_16",
    "SQG_WAVES_RESTORED",
    "SQG_WAVES_SAVED",
    "SQG_WAVES_INITIAL_PREFETCH",
    "SQG_NONE",
    "SQG_ACCUM_PREV",
    "SQG_EVENTS",
]


def test_validate_sqg_counter_collection_csv_pmc3(input_csv_pmc3: pd.DataFrame):
    _validate_csv_shape(input_csv_pmc3, KERNEL_LIST)
    _validate_csv_counter_set(input_csv_pmc3, PMC3_COUNTERS)
    _validate_csv_values(input_csv_pmc3, PMC3_COUNTERS)


def test_validate_sqg_counter_collection_json_pmc3(input_json_pmc3):
    _validate_json_shape(input_json_pmc3, PMC3_COUNTERS)


# ----------------------------------------------------------------------
# pmc4: graphics-only counters (export buses, message bus). Compute-only
# workload may legitimately leave these at 0; we only check plumbing/shape.
# ----------------------------------------------------------------------


PMC4_COUNTERS = [
    "SQG_PS_QUADS",
    "SQG_EXP_BUS0_BUSY",
    "SQG_EXP_BUS1_BUSY",
    "SQG_EXP_REQ0_BUS_BUSY",
    "SQG_EXP_REQ1_BUS_BUSY",
    "SQG_MSG",
    "SQG_MSG_BUS_BUSY",
    "SQG_MSG_INTERRUPT",
]


def test_validate_sqg_counter_collection_csv_pmc4(input_csv_pmc4: pd.DataFrame):
    _validate_csv_shape(input_csv_pmc4, KERNEL_LIST)
    _validate_csv_counter_set(input_csv_pmc4, PMC4_COUNTERS)
    _validate_csv_values(input_csv_pmc4, PMC4_COUNTERS)


def test_validate_sqg_counter_collection_json_pmc4(input_json_pmc4):
    _validate_json_shape(input_json_pmc4, PMC4_COUNTERS)


# ----------------------------------------------------------------------
# pmc5: thread-trace counters. ATT is not enabled in this run so all four
# may legitimately be 0; only structure/shape is validated.
# ----------------------------------------------------------------------


PMC5_COUNTERS = [
    "SQG_TTRACE_REQS",
    "SQG_TTRACE_INFLIGHT_REQS",
    "SQG_TTRACE_STALL",
    "SQG_TTRACE_LOST_PACKETS",
]


def test_validate_sqg_counter_collection_csv_pmc5(input_csv_pmc5: pd.DataFrame):
    _validate_csv_shape(input_csv_pmc5, KERNEL_LIST)
    _validate_csv_counter_set(input_csv_pmc5, PMC5_COUNTERS)
    _validate_csv_values(input_csv_pmc5, PMC5_COUNTERS)


def test_validate_sqg_counter_collection_json_pmc5(input_json_pmc5):
    _validate_json_shape(input_json_pmc5, PMC5_COUNTERS)


def test_validate_sqg_counter_classification_disjoint():
    """Sanity: in-test classification of SQG counters does not overlap."""
    overlap = ALWAYS_NONZERO & MUST_BE_ZERO
    assert (
        not overlap
    ), f"Counters in both ALWAYS_NONZERO and MUST_BE_ZERO: {sorted(overlap)}"

    wave_pair_set = {name for pair in WAVE_SPLIT_PAIRS for name in pair}
    overlap_pairs = wave_pair_set & (ALWAYS_NONZERO | MUST_BE_ZERO)
    assert (
        not overlap_pairs
    ), f"Wave-split pair counter also classified elsewhere: {sorted(overlap_pairs)}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
