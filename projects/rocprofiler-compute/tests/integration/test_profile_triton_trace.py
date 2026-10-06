# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for Triton operator tracing during profiling."""

import csv
import re
import sys
from pathlib import Path

import common
import pytest

from tests.integration.common import config, require_triton
from utils import csv_compression


@pytest.fixture(scope="module")
def triton_trace_workload_state():
    state = {"dir": None, "profiled": False}
    yield state
    if state["dir"] is not None:
        common.clean_output_dir(config["cleanup"], state["dir"])


@pytest.fixture
def triton_trace_profiled_workload(
    triton_trace_workload_state,
    binary_handler_profile_rocprof_compute,
):
    require_triton(gpu=True)
    if not triton_trace_workload_state["profiled"]:
        workload_dir = common.get_output_dir(param_id="triton_trace")
        triton_trace_workload_state["dir"] = workload_dir
        profile_config = dict(config)
        profile_config["triton_test_app"] = [
            sys.executable,
            "./sample/triton_ffn.py",
        ]
        returncode = binary_handler_profile_rocprof_compute(
            profile_config,
            workload_dir,
            [
                "--experimental",
                "--triton-trace",
                "--iteration-multiplexing",
            ],
            check_success=True,
            app_name="triton_test_app",
        )
        assert returncode == 0, "Profiling the triton application failed"
        triton_trace_workload_state["profiled"] = True
    return triton_trace_workload_state["dir"]


@pytest.mark.triton_trace
def test_triton_trace_profile_csvs(triton_trace_profiled_workload):
    workload_dir = triton_trace_profiled_workload
    marker_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv.gz"))
    assert marker_files, "No marker_api_trace.csv.gz produced"
    functions = []
    for marker_file in marker_files:
        corresponding_counter_file = marker_file.parent / marker_file.name.replace(
            "marker_api_trace", "counter_collection"
        )
        assert corresponding_counter_file.is_file(), (
            f"counter_collection CSV not found for {marker_file}"
        )
        with csv_compression.open_gzip_csv_read(marker_file) as f:
            for row in csv.DictReader(f):
                functions.append(row["Function"])
    assert any("|triton" in fn for fn in functions)
    assert any("triton.JITFunction.matmul_kernel" in fn for fn in functions)


@pytest.mark.triton_trace
def test_list_triton_operators_prints_call_tree(
    triton_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-triton-operators",
        "--path",
        triton_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Triton Operator Call Tree:" in out
    assert "triton.JITFunction.matmul_kernel" in out


@pytest.mark.triton_trace
def test_triton_operator_matmul_selects_kernels(
    triton_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--triton-operator",
        "*matmul*",
        "--path",
        triton_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Matched Triton Operators:" in out
    assert "triton.JITFunction.matmul_kernel" in out


@pytest.mark.triton_trace
def test_triton_operator_intersects_kernel_id(
    triton_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    list_code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-triton-operators",
        "--path",
        triton_trace_profiled_workload,
    ])
    assert list_code == 0
    captured = capsys.readouterr()
    list_out = captured.out + captured.err
    matmul_idx = list_out.find("triton.JITFunction.matmul_kernel")
    assert matmul_idx >= 0
    match = re.search(r"\(id (\d+)\)", list_out[matmul_idx:])
    assert match is not None
    kernel_id = match.group(1)
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--triton-operator",
        "*matmul*",
        "--kernel",
        kernel_id,
        "--path",
        triton_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "operator filter selected 1 kernel" in out
