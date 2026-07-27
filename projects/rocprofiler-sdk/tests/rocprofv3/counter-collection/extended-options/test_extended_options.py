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

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

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
SANITIZED_ENVIRONMENT_VARIABLES = (
    "ROCPROFILER_CI",
    "ROCPROF_MPI_RANKS",
    "ROCPROF_MPI_RANK_VAR",
    "ROCPROF_MPI_SIZE_VAR",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "SLURM_PROCID",
    "SLURM_NTASKS",
)


def _environment():
    environment = os.environ.copy()
    for key in SANITIZED_ENVIRONMENT_VARIABLES:
        environment.pop(key, None)
    return environment


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        assert set(reader.fieldnames or ()) == COUNTER_COLUMNS
        rows = list(reader)
    assert rows
    return rows


def _validate_counter_rows(rows):
    assert {row["Counter_Name"] for row in rows} == {"SQ_WAVES"}
    dispatch_ids = [int(row["Dispatch_Id"]) for row in rows]
    assert len(dispatch_ids) == len(set(dispatch_ids))
    assert sorted(dispatch_ids) == list(range(1, len(dispatch_ids) + 1))
    assert all(float(row["Counter_Value"]) >= 0.0 for row in rows)
    assert any(float(row["Counter_Value"]) > 0.0 for row in rows)

    by_agent = defaultdict(list)
    for row in rows:
        start = int(row["Start_Timestamp"])
        end = int(row["End_Timestamp"])
        assert end >= start
        by_agent[row["Agent_Id"]].append((start, end))
    for intervals in by_agent.values():
        intervals.sort()
        assert all(
            next_start >= end
            for (_, end), (next_start, _) in zip(intervals, intervals[1:])
        )


def _run_profile(
    workloads,
    tmp_path,
    case,
    application,
    tool_args=(),
    application_args=(),
    formats=("csv",),
):
    output_directory = tmp_path / case
    command = [
        workloads["rocprofv3"],
        "--pmc",
        "SQ_WAVES",
    ]
    command.extend(tool_args)
    command.extend(
        [
            "-d",
            str(output_directory),
            "-o",
            case,
            "--output-format",
        ]
    )
    command.extend(formats)
    command.extend(["--", application])
    command.extend(str(argument) for argument in application_args)

    result = subprocess.run(
        command,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr

    csv_files = sorted(output_directory.rglob("*_counter_collection.csv"))
    assert len(csv_files) == 1
    rows = _read_csv(csv_files[0])
    _validate_counter_rows(rows)

    json_files = sorted(output_directory.rglob("*_results.json"))
    json_data = None
    if "json" in formats:
        assert len(json_files) == 1
        with json_files[0].open(encoding="utf-8") as input_file:
            json_data = json.load(input_file)
    if "pftrace" in formats:
        pftrace_files = sorted(output_directory.rglob("*.pftrace"))
        assert len(pftrace_files) == 1
        assert pftrace_files[0].stat().st_size > 0
    return result, rows, json_data


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


def test_kernel_name_modes(workloads, tmp_path):
    _, reference, _ = _run_profile(
        workloads, tmp_path, "names-reference", workloads["vector_ops"]
    )
    _, truncated, _ = _run_profile(
        workloads,
        tmp_path,
        "names-truncated",
        workloads["vector_ops"],
        tool_args=("-T",),
    )
    _, mangled, _ = _run_profile(
        workloads,
        tmp_path,
        "names-mangled",
        workloads["vector_ops"],
        tool_args=("-M",),
    )

    reference_names = {row["Kernel_Name"] for row in _application_rows(reference)}
    truncated_names = {row["Kernel_Name"] for row in _application_rows(truncated)}
    mangled_names = {row["Kernel_Name"] for row in _application_rows(mangled)}
    assert reference_names and truncated_names and mangled_names
    assert any("(" in name or "<" in name for name in reference_names)
    assert all("(" not in name and "<" not in name for name in truncated_names)
    assert all(name.startswith("_Z") for name in mangled_names)

    expected_count = len(_application_rows(reference))
    assert len(_application_rows(truncated)) == expected_count
    assert len(_application_rows(mangled)) == expected_count


@pytest.mark.parametrize(
    "agent_index,agent_field",
    (
        ("absolute", "node_id"),
        ("relative", "logical_node_id"),
        ("type-relative", "logical_node_type_id"),
    ),
)
def test_agent_index_modes(workloads, tmp_path, agent_index, agent_field):
    _, rows, json_data = _run_profile(
        workloads,
        tmp_path,
        "agent-index-{}".format(agent_index),
        workloads["vector_ops"],
        tool_args=("--agent-index", agent_index),
        formats=("csv", "json"),
    )
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
    for row in rows:
        assert (
            int(row["Agent_Id"].split()[-1])
            == expected_by_dispatch[int(row["Dispatch_Id"])]
        )


def test_queue_thread_gpu_and_kernel_rename_contexts(workloads, tmp_path):
    application = workloads["reproducible"]
    if not application:
        pytest.skip("reproducible-dispatch-count is unavailable")

    _, rows, json_data = _run_profile(
        workloads,
        tmp_path,
        "execution-contexts",
        application,
        tool_args=("--group-by-queue", "--kernel-rename", "--marker-trace"),
        application_args=(200, 2),
        formats=("csv", "json", "pftrace"),
    )
    application_data = _application_rows(rows)
    assert len({row["Queue_Id"] for row in application_data}) >= 2
    assert len({row["Thread_Id"] for row in application_data}) >= 2
    assert "iteration" in {row["Kernel_Name"] for row in application_data}

    gpu_agents = [
        agent for agent in _tool_data(json_data)["agents"] if int(agent["type"]) == 2
    ]
    if len(gpu_agents) >= 2:
        assert len({row["Agent_Id"] for row in application_data}) >= 2


@pytest.mark.parametrize(
    "case,tool_args",
    (
        pytest.param("stats", ("--stats",), id="stats"),
        pytest.param(
            "summary",
            ("--summary", "--summary-per-domain"),
            id="summary",
        ),
    ),
)
def test_counter_collection_with_post_processing(workloads, tmp_path, case, tool_args):
    result, _, _ = _run_profile(
        workloads,
        tmp_path,
        case,
        workloads["vector_ops"],
        tool_args=tool_args,
    )
    assert "COUNTER_COLLECTION" not in result.stdout + result.stderr


def test_openmp_counter_collection_when_available(workloads, tmp_path):
    if not workloads["openmp"]:
        pytest.skip("OpenMP offload workload is unavailable")
    _, rows, _ = _run_profile(workloads, tmp_path, "openmp", workloads["openmp"])
    assert _application_rows(rows)


def test_pytorch_counter_collection_when_available(workloads, tmp_path):
    availability = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available()",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if availability.returncode != 0:
        pytest.skip("ROCm PyTorch is unavailable")

    script = tmp_path / "pytorch_workload.py"
    script.write_text(
        "import torch\n"
        "a = torch.ones((512, 512), device='cuda')\n"
        "torch.mm(a, a)\n"
        "torch.cuda.synchronize()\n",
        encoding="utf-8",
    )
    _, rows, _ = _run_profile(
        workloads,
        tmp_path,
        "pytorch",
        sys.executable,
        application_args=(script,),
    )
    assert _application_rows(rows)


def test_invalid_counter_reports_product_gap(workloads, tmp_path):
    output_directory = tmp_path / "invalid-counter"
    invalid_counter = "__ROCPROF_INVALID_COUNTER__"
    result = subprocess.run(
        [
            workloads["rocprofv3"],
            "--pmc",
            invalid_counter,
            "-d",
            str(output_directory),
            "-o",
            "invalid",
            "--",
            workloads["vector_ops"],
        ],
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert invalid_counter in output
    assert not list(output_directory.rglob("*_counter_collection.csv"))
    if result.returncode == 0:
        pytest.xfail("rocprofv3 currently exits successfully for an unknown counter")
    assert result.returncode > 0
