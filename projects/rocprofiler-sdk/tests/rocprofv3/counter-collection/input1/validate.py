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

import math
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import pytest

kernel_list = sorted(
    ["addition_kernel", "subtract_kernel", "multiply_kernel", "divide_kernel"]
)
REQUESTED_COUNTER = "SQ_WAVES"
INTERNAL_KERNEL_PATTERN = re.compile(r"__amd_rocclr_")

# CSV uses six fixed decimals for values at least one and eight scientific decimals below
# one. The relative tolerance covers summing dimension values at larger magnitudes.
COUNTER_VALUE_ABS_TOLERANCE = 1.0e-6
COUNTER_VALUE_REL_TOLERANCE = 1.0e-12

# AQLProfile currently reports unreliable counter values for these architectures.
VALUE_CHECK_SKIP_GFX = {
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
}


def _is_internal_kernel(kernel_name):
    return INTERNAL_KERNEL_PATTERN.search(kernel_name) is not None


def _integer_column(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    assert values.notna().all(), f"{column} contains a non-numeric value"
    integer_values = values.astype("int64")
    assert (values == integer_values).all(), f"{column} contains a non-integer value"
    return integer_values


def _normalize_csv_dispatches(df):
    assert not df.empty

    agent_ids = pd.to_numeric(
        df["Agent_Id"].astype(str).str.rsplit(" ", n=1).str[-1], errors="coerce"
    )
    assert agent_ids.notna().all()
    assert (agent_ids >= 0).all()
    assert (_integer_column(df, "Queue_Id") > 0).all()
    assert (_integer_column(df, "Process_Id") > 0).all()
    assert df["Kernel_Name"].notna().all()
    assert df["Kernel_Name"].str.len().gt(0).all()
    counter_values = pd.to_numeric(df["Counter_Value"], errors="coerce")
    assert counter_values.notna().all()

    normalized = pd.DataFrame(
        {
            "dispatch_id": _integer_column(df, "Dispatch_Id"),
            "kernel_id": _integer_column(df, "Kernel_Id"),
            "kernel_name": df["Kernel_Name"],
            "counter_name": df["Counter_Name"],
            "counter_value": counter_values,
        }
    )

    identities = []
    for dispatch_id, rows in normalized.groupby("dispatch_id", sort=True):
        kernel_identity = (
            int(rows.iloc[0]["kernel_id"]),
            rows.iloc[0]["kernel_name"],
        )
        observed_kernels = list(zip(rows["kernel_id"], rows["kernel_name"]))
        assert all(
            (int(kernel_id), kernel_name) == kernel_identity
            for kernel_id, kernel_name in observed_kernels
        ), f"dispatch {dispatch_id} contains mixed kernel identities"

        counter_names = rows["counter_name"].tolist()
        assert counter_names and all(
            counter_name == REQUESTED_COUNTER for counter_name in counter_names
        ), f"dispatch {dispatch_id} contains an unexpected counter: {counter_names}"
        assert len(rows) == 1, (
            f"dispatch {dispatch_id} contains duplicate "
            f"{kernel_identity[1]}/{REQUESTED_COUNTER} records"
        )
        counter_value = float(rows.iloc[0]["counter_value"])

        identities.append(
            (
                int(dispatch_id),
                kernel_identity[0],
                kernel_identity[1],
                REQUESTED_COUNTER,
                counter_value,
            )
        )

    dispatch_ids = [identity[0] for identity in identities]
    assert dispatch_ids == list(range(1, len(dispatch_ids) + 1))
    return tuple(identities)


def _metadata_by_handle(entries):
    return {int(entry["id"]["handle"]): entry for entry in entries}


def _normalize_json_dispatches(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    assert counter_collection_data

    agents = _metadata_by_handle(data["agents"])
    counters = _metadata_by_handle(data["counters"])
    dispatch_groups = defaultdict(list)

    for collection in counter_collection_data:
        dispatch_data = collection["dispatch_data"]["dispatch_info"]
        dispatch_id = int(dispatch_data["dispatch_id"])
        agent_id = int(dispatch_data["agent_id"]["handle"])
        queue_id = int(dispatch_data["queue_id"]["handle"])
        kernel_id = int(dispatch_data["kernel_id"])

        assert dispatch_id > 0
        assert agent_id > 0
        assert queue_id > 0
        assert agent_id in agents, f"dispatch {dispatch_id} references unknown agent"

        kernel_name = data["kernel_symbols"][kernel_id]["formatted_kernel_name"]
        assert kernel_name

        records = collection["records"]
        assert records, f"dispatch {dispatch_id} has no counter records"

        counter_names = []
        counter_values = []
        for record in records:
            counter_id = int(record["counter_id"]["handle"])
            counter = counters.get(counter_id)
            assert counter is not None, f"record references unknown counter: {record}"
            counter_names.append(counter["name"])
            counter_values.append(float(record["value"]))

        dispatch_groups[dispatch_id].append(
            {
                "kernel_identity": (kernel_id, kernel_name),
                "counter_names": counter_names,
                "counter_value": math.fsum(counter_values),
                "architecture": agents[agent_id]["name"],
            }
        )

    identities = []
    for dispatch_id in sorted(dispatch_groups):
        entries = dispatch_groups[dispatch_id]
        kernel_identity = entries[0]["kernel_identity"]
        assert all(
            entry["kernel_identity"] == kernel_identity for entry in entries
        ), f"dispatch {dispatch_id} contains mixed kernel identities"

        counter_names = [
            counter_name for entry in entries for counter_name in entry["counter_names"]
        ]
        assert counter_names and all(
            counter_name == REQUESTED_COUNTER for counter_name in counter_names
        ), f"dispatch {dispatch_id} contains an unexpected counter: {counter_names}"
        assert len(entries) == 1, (
            f"dispatch {dispatch_id} contains duplicate "
            f"{kernel_identity[1]}/{REQUESTED_COUNTER} records"
        )

        entry = entries[0]
        if (
            not _is_internal_kernel(kernel_identity[1])
            and entry["architecture"] not in VALUE_CHECK_SKIP_GFX
        ):
            assert entry["counter_value"] > 0, f"dispatch {dispatch_id} has no SQ_WAVES"

        identities.append(
            (
                dispatch_id,
                kernel_identity[0],
                kernel_identity[1],
                REQUESTED_COUNTER,
                entry["counter_value"],
            )
        )

    dispatch_ids = [identity[0] for identity in identities]
    assert dispatch_ids == list(range(1, len(dispatch_ids) + 1))
    return tuple(identities)


def test_validate_counter_collection_pmc1(
    input_data: pd.DataFrame,
    json_data,
):
    csv_identities = _normalize_csv_dispatches(input_data)
    json_identities = _normalize_json_dispatches(json_data)

    csv_structure = tuple(dispatch[:-1] for dispatch in csv_identities)
    json_structure = tuple(dispatch[:-1] for dispatch in json_identities)
    assert csv_structure == json_structure, (
        "CSV and JSON dispatch/kernel/counter identities differ:\n"
        f"CSV: {csv_structure}\nJSON: {json_structure}"
    )

    for csv_dispatch, json_dispatch in zip(csv_identities, json_identities):
        identity = csv_dispatch[:-1]
        csv_value = csv_dispatch[-1]
        json_value = json_dispatch[-1]
        assert csv_value == pytest.approx(
            json_value,
            rel=COUNTER_VALUE_REL_TOLERANCE,
            abs=COUNTER_VALUE_ABS_TOLERANCE,
        ), (
            f"CSV and JSON counter values differ for {identity}: "
            f"CSV={csv_value}, JSON sum={json_value}"
        )

    kernel_counts = Counter(
        kernel_name
        for _, _, kernel_name, _, _ in csv_identities
        if not _is_internal_kernel(kernel_name)
    )
    assert sorted(kernel_counts) == kernel_list
    counts = list(kernel_counts.values())
    assert counts and min(counts) == max(counts)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
