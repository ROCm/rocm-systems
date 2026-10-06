#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from collections import defaultdict
from pathlib import Path
import sys

import pytest

COUNTER_COLUMNS = {
    "Correlation_Id",
    "Dispatch_Id",
    "Agent_Id",
    "Queue_Id",
    "Process_Id",
    "Thread_Id",
    "Grid_Size",
    "Kernel_Id",
    "Kernel_Name",
    "Workgroup_Size",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Counter_Name",
    "Counter_Value",
    "Start_Timestamp",
    "End_Timestamp",
}


def _validate_counter_rows(rows):
    assert rows
    assert set(rows[0]) == COUNTER_COLUMNS
    assert {row["Counter_Name"] for row in rows} == {"SQ_WAVES"}

    dispatch_ids = [int(row["Dispatch_Id"]) for row in rows]
    assert len(dispatch_ids) == len(set(dispatch_ids))
    assert sorted(dispatch_ids) == list(range(1, len(dispatch_ids) + 1))
    assert all(float(row["Counter_Value"]) >= 0.0 for row in rows)
    assert any(float(row["Counter_Value"]) > 0.0 for row in rows)

    # counter collection serializes kernels, so dispatches on one agent never overlap.
    # Keying on dispatch id keeps multi-counter runs to one interval per dispatch.
    intervals_by_agent = defaultdict(dict)
    for row in rows:
        intervals_by_agent[row["Agent_Id"]][int(row["Dispatch_Id"])] = (
            int(row["Start_Timestamp"]),
            int(row["End_Timestamp"]),
        )
    for intervals in intervals_by_agent.values():
        ordered = sorted(intervals.values())
        assert all(
            next_start >= end for (_, end), (next_start, _) in zip(ordered, ordered[1:])
        )


def _application_rows(rows):
    return [
        row
        for row in rows
        if row["Kernel_Name"] and not row["Kernel_Name"].startswith("__amd_rocclr_")
    ]


def _tool_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    if isinstance(data, list):
        assert len(data) == 1
        return data[0]
    return data


def test_kernel_name_modes(reference_rows, truncated_rows, mangled_rows):
    for rows in (reference_rows, truncated_rows, mangled_rows):
        _validate_counter_rows(rows)

    reference_names = {row["Kernel_Name"] for row in _application_rows(reference_rows)}
    truncated_names = {row["Kernel_Name"] for row in _application_rows(truncated_rows)}
    mangled_names = {row["Kernel_Name"] for row in _application_rows(mangled_rows)}
    assert reference_names and truncated_names and mangled_names
    assert any("(" in name or "<" in name for name in reference_names)
    assert all("(" not in name and "<" not in name for name in truncated_names)
    assert all(name.startswith("_Z") for name in mangled_names)

    expected_count = len(_application_rows(reference_rows))
    assert len(_application_rows(truncated_rows)) == expected_count
    assert len(_application_rows(mangled_rows)) == expected_count


def test_agent_index_mode(counter_rows, json_data, agent_index):
    _validate_counter_rows(counter_rows)
    agent_field = {
        "absolute": "node_id",
        "relative": "logical_node_id",
        "type-relative": "logical_node_type_id",
    }[agent_index]

    data = _tool_data(json_data)
    agents = {agent["id"]["handle"]: agent for agent in data["agents"]}
    expected_by_dispatch = {
        int(entry["dispatch_data"]["dispatch_info"]["dispatch_id"]): int(
            agents[entry["dispatch_data"]["dispatch_info"]["agent_id"]["handle"]][
                agent_field
            ]
        )
        for entry in data["callback_records"]["counter_collection"]
    }
    assert expected_by_dispatch
    for row in counter_rows:
        assert (
            int(row["Agent_Id"].split()[-1])
            == expected_by_dispatch[int(row["Dispatch_Id"])]
        )


def test_execution_contexts(context_rows, context_json, context_pftrace):
    _validate_counter_rows(context_rows)
    application_rows = _application_rows(context_rows)
    assert len({row["Queue_Id"] for row in application_rows}) >= 2
    assert len({row["Thread_Id"] for row in application_rows}) >= 2
    assert "iteration" in {row["Kernel_Name"] for row in application_rows}

    visible_gpu_agents = [
        agent
        for agent in _tool_data(context_json)["agents"]
        if int(agent["type"]) == 2
        and agent["runtime_visibility"]["hsa"]
        and agent["runtime_visibility"]["hip"]
    ]
    if len(visible_gpu_agents) >= 2:
        assert len({row["Agent_Id"] for row in application_rows}) >= 2

    pftrace = Path(context_pftrace)
    assert pftrace.is_file()
    assert pftrace.stat().st_size > 0


def test_counter_collection_with_post_processing(post_rows):
    _validate_counter_rows(post_rows)


def test_openmp_counter_collection(openmp_rows):
    _validate_counter_rows(openmp_rows)
    assert _application_rows(openmp_rows)


def test_device_qualifier_restricts_counter(device_rows, json_data):
    assert device_rows

    # the run pins --agent-index relative, so Agent_Id holds the logical node id
    data = _tool_data(json_data)
    gpu_index_by_agent = {
        int(agent["logical_node_id"]): int(agent["gpu_index"])
        for agent in data["agents"]
        if int(agent["type"]) == 2
    }
    names_by_gpu_index = defaultdict(set)
    for row in device_rows:
        gpu_index = gpu_index_by_agent[int(row["Agent_Id"].split()[-1])]
        names_by_gpu_index[gpu_index].add(row["Counter_Name"])

    qualified_agent_is_visible = any(
        int(agent["gpu_index"]) == 0
        and agent["runtime_visibility"]["hsa"]
        and agent["runtime_visibility"]["hip"]
        for agent in data["agents"]
        if int(agent["type"]) == 2
    )
    if not qualified_agent_is_visible:
        pytest.skip(
            "GPU index 0 is not visible to the runtime, so :device=0 collects nothing"
        )

    # GRBM_GUI_ACTIVE is requested with :device=0, SQ_WAVES without a qualifier
    for gpu_index, names in names_by_gpu_index.items():
        assert ("GRBM_GUI_ACTIVE" in names) == (gpu_index == 0)
        assert "SQ_WAVES" in names

    qualified = [row for row in device_rows if row["Counter_Name"] == "GRBM_GUI_ACTIVE"]
    assert all(float(row["Counter_Value"]) >= 0.0 for row in qualified)
    assert any(float(row["Counter_Value"]) > 0.0 for row in qualified)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
