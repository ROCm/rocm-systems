# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for combined ML API tracing during profiling."""

import csv
import sys
from pathlib import Path

import common
import pytest

from tests.integration.common import config, require_triton
from utils import csv_compression


@pytest.fixture(scope="module")
def ml_api_trace_workload_state():
    state = {"dir": None, "profiled": False}
    yield state
    if state["dir"] is not None:
        common.clean_output_dir(config["cleanup"], state["dir"])


@pytest.fixture
def ml_api_trace_profiled_workload(
    ml_api_trace_workload_state,
    binary_handler_profile_rocprof_compute,
):
    require_triton(gpu=True)
    if not ml_api_trace_workload_state["profiled"]:
        workload_dir = common.get_output_dir(param_id="ml_api_trace")
        ml_api_trace_workload_state["dir"] = workload_dir
        profile_config = dict(config)
        profile_config["torch_compile_test_app"] = [
            sys.executable,
            *config["torch_compile_test_app"][1:],
        ]
        returncode = binary_handler_profile_rocprof_compute(
            profile_config,
            workload_dir,
            [
                "--experimental",
                "--ml-api-trace",
                "--iteration-multiplexing",
            ],
            check_success=True,
            app_name="torch_compile_test_app",
        )
        assert returncode == 0, "Profiling the ml-api application failed"
        ml_api_trace_workload_state["profiled"] = True
    return ml_api_trace_workload_state["dir"]


@pytest.mark.ml_api_trace
def test_ml_api_trace_profile_csvs(ml_api_trace_profiled_workload):
    marker_files = list(
        Path(ml_api_trace_profiled_workload).glob("**/*marker_api_trace.csv.gz")
    )
    assert marker_files, "No marker_api_trace.csv.gz produced"
    functions = []
    for marker_file in marker_files:
        with csv_compression.open_gzip_csv_read(marker_file) as f:
            for row in csv.DictReader(f):
                functions.append(row["Function"])
    assert any("|torch" in fn for fn in functions)
    assert any("|triton" in fn for fn in functions)


@pytest.mark.ml_api_trace
def test_list_both_operators(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--list-triton-operators",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "PyTorch, Triton Operator Call Tree:" in out
    assert "aten::relu" in out
    assert "triton.JITFunction" in out
