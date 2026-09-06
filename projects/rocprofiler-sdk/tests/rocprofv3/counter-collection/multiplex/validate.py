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

from collections import defaultdict


def test_agent_info(agent_info_input_data):
    logical_node_id = max([int(itr["Logical_Node_Id"]) for itr in agent_info_input_data])

    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


def expected_group_schedule(num_dispatches, pmc_group_interval, num_groups):
    """The group each dispatch is scheduled under. The tool advances a counter once
    per profiled dispatch on a device and selects (counter / interval) % num_groups,
    so the schedule follows each device's own dispatch ordering."""
    return [(idx // pmc_group_interval) % num_groups for idx in range(num_dispatches)]


def _matching_group_indices(seen_counters, group_counters):
    return [
        group_id
        for group_id, counters in enumerate(group_counters)
        if seen_counters == counters
    ]


def _collection_schedule(counter_input_data):
    schedule = defaultdict(lambda: defaultdict(set))
    for row in counter_input_data:
        agent_id = row["Agent_Id"]
        dispatch_id = int(row["Dispatch_Id"])
        counter_name = row["Counter_Name"]
        dispatch_counters = schedule[agent_id][dispatch_id]
        assert counter_name not in dispatch_counters, (
            f"duplicate counter row for agent {agent_id}, dispatch {dispatch_id}, "
            f"counter {counter_name}"
        )
        dispatch_counters.add(counter_name)
    return schedule


def test_dispatch_accounting(counter_input_data, expected_dispatch_count):
    per_agent = _collection_schedule(counter_input_data)
    assert per_agent, "no counter collection data was produced"

    all_dispatch_ids = sorted(
        dispatch_id for dispatches in per_agent.values() for dispatch_id in dispatches
    )
    assert all_dispatch_ids == list(
        range(1, expected_dispatch_count + 1)
    ), f"expected dispatch ids 1..{expected_dispatch_count}, got {all_dispatch_ids}"


def test_counter_collection_multiplex(
    counter_input_data,
    multiplex_layout,
    allow_zero_counter_values,
):
    assert (
        multiplex_layout is not None
    ), "--multiplex-input is required to validate the group schedule"

    pmc_groups, pmc_group_interval = multiplex_layout
    num_groups = len(pmc_groups)

    group_counters = [set(group) for group in pmc_groups]
    all_counters = set().union(*group_counters)

    for row in counter_input_data:
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0

        assert len(row["Counter_Value"]) > 0
        assert row["Counter_Name"] in all_counters
        if allow_zero_counter_values:
            assert float(row["Counter_Value"]) >= 0
        else:
            assert float(row["Counter_Value"]) > 0

    per_agent = _collection_schedule(counter_input_data)
    observed_group_counters = [set() for _ in pmc_groups]

    for agent_id, dispatch_counters in per_agent.items():
        dispatch_ids = sorted(dispatch_counters)
        expected_groups = expected_group_schedule(
            len(dispatch_ids), pmc_group_interval, num_groups
        )

        for position, (dispatch_id, group_id) in enumerate(
            zip(dispatch_ids, expected_groups)
        ):
            seen_counters = dispatch_counters[dispatch_id]

            # the collected set is exactly one group of the layout: no counter from
            # another group leaked in, and no counter of this group is missing
            matching = _matching_group_indices(seen_counters, group_counters)
            assert matching, (
                f"agent {agent_id} dispatch {dispatch_id} collected "
                f"{sorted(seen_counters)}, which is not exactly any group of "
                f"{[sorted(group) for group in group_counters]}"
            )

            # ... and it is the group this position in the rotation is scheduled for
            assert group_id in matching, (
                f"agent {agent_id} dispatch {dispatch_id} (dispatch "
                f"{position + 1} on this device) is scheduled for group {group_id} "
                f"({sorted(group_counters[group_id])}) but collected group "
                f"{matching[0]} ({sorted(seen_counters)})"
            )

            observed_group_counters[group_id] |= seen_counters

    for group_id, expected_counters in enumerate(group_counters):
        assert observed_group_counters[group_id] == expected_counters, (
            f"group {group_id} ({sorted(expected_counters)}) was not fully "
            f"collected, saw {sorted(observed_group_counters[group_id])}"
        )


def test_counter_value_stability(counter_input_data, stable_counters, max_value_ratio):
    """Values for a counter across identical repeated dispatches must be stable
    (max <= ratio * min). Restricted to --stable-counters: only counters that measure
    the kernel itself qualify, since free-running clock counters such as GRBM_COUNT
    time the whole dispatch window and vary with unrelated load on the device."""
    if stable_counters is None:
        pytest.skip("--stable-counters not set (workload not identical-dispatch)")

    by_counter = defaultdict(list)
    for row in counter_input_data:
        if row["Counter_Name"] not in stable_counters:
            continue
        by_counter[(row["Agent_Id"], row["Counter_Name"])].append(
            float(row["Counter_Value"])
        )

    assert {counter_name for _, counter_name in by_counter} == stable_counters, (
        f"--stable-counters {sorted(stable_counters)} were not all collected, saw "
        f"{sorted({counter_name for _, counter_name in by_counter})}"
    )

    checked = 0
    for (agent_id, counter_name), values in by_counter.items():
        assert all(v > 0 for v in values), f"{counter_name} has a non-positive value"
        if len(values) < 2:
            continue
        checked += 1
        lo, hi = min(values), max(values)
        assert hi <= max_value_ratio * lo, (
            f"agent {agent_id} counter {counter_name} varies too much across "
            f"identical dispatches: min={lo}, max={hi} (ratio "
            f"{hi / lo:.2f} > {max_value_ratio})"
        )

    assert checked > 0, "no counter had repeated dispatches to check stability"


def test_run_level_json_yaml_equivalence(counter_input_data, counter_input_b_data):
    """The same layout run from JSON and from YAML must collect the same counters
    per dispatch per device (schedule only; values are run-to-run noisy). Skipped
    unless --counter-input-b is provided."""
    if counter_input_b_data is None:
        pytest.skip("--counter-input-b not set (single-format run)")

    a = _collection_schedule(counter_input_data)
    b = _collection_schedule(counter_input_b_data)

    assert a, "primary run produced no counter data"
    assert b, "comparison run produced no counter data"
    assert set(a) == set(
        b
    ), f"different devices collected: {sorted(a)} (JSON) vs {sorted(b)} (YAML)"

    for agent_id in a:
        dispatches_a, dispatches_b = a[agent_id], b[agent_id]
        assert set(dispatches_a) == set(dispatches_b), (
            f"agent {agent_id} covered different dispatches: "
            f"{sorted(dispatches_a)} (JSON) vs {sorted(dispatches_b)} (YAML)"
        )
        for dispatch_id in dispatches_a:
            assert dispatches_a[dispatch_id] == dispatches_b[dispatch_id], (
                f"agent {agent_id} dispatch {dispatch_id} collected "
                f"{sorted(dispatches_a[dispatch_id])} (JSON) but "
                f"{sorted(dispatches_b[dispatch_id])} (YAML)"
            )


def test_graceful_degradation_drops_unresolved_groups(
    counter_input_data, present_counters, allow_zero_counter_values
):
    """An unresolvable (unknown counter) or empty group is dropped while the valid
    group is still collected in full. Asserts every dispatch contains exactly
    --present-counters. Skipped unless --present-counters is set."""
    if present_counters is None:
        pytest.skip("--present-counters not set (not a graceful-degradation run)")

    for row in counter_input_data:
        name = row["Counter_Name"]
        assert name in present_counters, (
            f"counter {name} was collected but is not in the expected surviving "
            f"set {sorted(present_counters)} (a dropped group leaked through)"
        )
        if allow_zero_counter_values:
            assert float(row["Counter_Value"]) >= 0
        else:
            assert float(row["Counter_Value"]) > 0

    for agent_id, dispatches in _collection_schedule(counter_input_data).items():
        for dispatch_id, counters in dispatches.items():
            assert counters == present_counters, (
                f"agent {agent_id} dispatch {dispatch_id} collected "
                f"{sorted(counters)}, expected surviving group "
                f"{sorted(present_counters)}"
            )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
