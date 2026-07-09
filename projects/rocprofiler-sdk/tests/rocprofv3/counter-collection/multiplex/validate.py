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


def expected_group_index(dispatch_id, pmc_group_interval, num_groups):
    return ((dispatch_id - 1) // pmc_group_interval) % num_groups


def _actual_group_index(seen_counters, group_counters):
    for group_id, counters in enumerate(group_counters):
        if seen_counters == counters:
            return group_id
    return None


def test_counter_collection_multiplex(
    counter_input_data, multiplex_layout, allow_zero_counter_values
):
    if multiplex_layout is None:
        pytest.skip("--multiplex-input not set (no single layout to validate)")

    pmc_groups, pmc_group_interval = multiplex_layout
    num_groups = len(pmc_groups)

    group_counters = [set(group) for group in pmc_groups]
    all_counters = set().union(*group_counters)

    # grouped per Agent_Id then Dispatch_Id (the interval rotates per device)
    per_agent = defaultdict(lambda: defaultdict(set))
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

        per_agent[row["Agent_Id"]][int(row["Dispatch_Id"])].add(row["Counter_Name"])

    assert per_agent, "no counter collection data was produced"

    observed_group_counters = [set() for _ in pmc_groups]

    for agent_id, dispatch_counters in per_agent.items():
        dispatch_ids = sorted(dispatch_counters)

        # Dispatch_Id must be contiguous 1..N (each dispatch profiled exactly once)
        assert dispatch_ids == list(
            range(1, len(dispatch_ids) + 1)
        ), f"agent {agent_id} dispatch ids are not contiguous from 1: {dispatch_ids}"

        group_sequence = []
        for dispatch_id in dispatch_ids:
            seen_counters = dispatch_counters[dispatch_id]
            group_id = expected_group_index(dispatch_id, pmc_group_interval, num_groups)
            expected_counters = group_counters[group_id]

            # a dispatch collects exactly its group's set (no leakage, no partial)
            assert seen_counters, f"dispatch {dispatch_id} collected no counters"
            assert seen_counters == expected_counters, (
                f"agent {agent_id} dispatch {dispatch_id} maps to group {group_id} "
                f"({sorted(expected_counters)}) but collected {sorted(seen_counters)}"
            )

            group_sequence.append(_actual_group_index(seen_counters, group_counters))
            observed_group_counters[group_id] |= seen_counters

        # consecutive dispatches stay on one group for exactly the interval
        # (the trailing run may be a partial cycle)
        run_start = 0
        for idx in range(1, len(group_sequence) + 1):
            if (
                idx == len(group_sequence)
                or group_sequence[idx] != group_sequence[run_start]
            ):
                run_len = idx - run_start
                if idx == len(group_sequence):
                    assert run_len <= pmc_group_interval, (
                        f"agent {agent_id}: trailing run of group "
                        f"{group_sequence[run_start]} has length {run_len} > "
                        f"interval {pmc_group_interval}"
                    )
                else:
                    assert run_len == pmc_group_interval, (
                        f"agent {agent_id}: run of group {group_sequence[run_start]} "
                        f"has length {run_len} != interval {pmc_group_interval}"
                    )
                run_start = idx

    for group_id, expected_counters in enumerate(group_counters):
        assert observed_group_counters[group_id] == expected_counters, (
            f"group {group_id} ({sorted(expected_counters)}) was not fully "
            f"collected, saw {sorted(observed_group_counters[group_id])}"
        )


def test_counter_value_stability(counter_input_data, max_value_ratio):
    """Values for a counter across identical repeated dispatches must be stable
    (max <= ratio * min). Skipped unless --max-value-ratio is set."""
    if max_value_ratio is None:
        pytest.skip("--max-value-ratio not set (workload not identical-dispatch)")

    by_counter = defaultdict(list)
    for row in counter_input_data:
        by_counter[(row["Agent_Id"], row["Counter_Name"])].append(
            float(row["Counter_Value"])
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


def _collection_schedule(counter_input_data):
    # {Agent_Id: {Dispatch_Id: {counter names}}}
    schedule = defaultdict(lambda: defaultdict(set))
    for row in counter_input_data:
        schedule[row["Agent_Id"]][int(row["Dispatch_Id"])].add(row["Counter_Name"])
    return schedule


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
    group is still collected in full. Asserts the collected set is exactly
    --present-counters. Skipped unless --present-counters is set."""
    if present_counters is None:
        pytest.skip("--present-counters not set (not a graceful-degradation run)")

    seen = set()
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
        seen.add(name)

    assert (
        seen == present_counters
    ), f"valid group not fully collected: missing {sorted(present_counters - seen)}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
