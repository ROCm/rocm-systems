# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for combined ML API tracing during profiling."""

import csv
import re
import sys
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration.common import config, require_triton
from utils import csv_compression
from utils.utils_analysis import simplify_kernel_name

pytestmark = pytest.mark.ml_api_trace


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
        profile_config["ml_api_test_app"] = [
            sys.executable,
            *config["ml_api_test_app"][1:],
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
            app_name="ml_api_test_app",
        )
        assert returncode == 0, "Profiling the ml-api application failed"
        ml_api_trace_workload_state["profiled"] = True
    return ml_api_trace_workload_state["dir"]


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


def test_list_torch_only_confines_tree_and_kernels(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "PyTorch Operator Call Tree:" in out
    assert "Triton Operator Call Tree" not in out
    assert "triton.JITFunction" not in out


def test_list_triton_only_confines_tree_and_kernels(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-triton-operators",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Triton Operator Call Tree:" in out
    assert "PyTorch Operator Call Tree" not in out
    relu_idx = out.find("aten::relu")
    triton_idx = out.find("triton.JITFunction")
    assert relu_idx >= 0
    assert triton_idx > relu_idx
    assert "(id " not in out[relu_idx:triton_idx]
    assert "(id " in out[triton_idx:]


def test_filter_triton_confines_metric_kernel_ids(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    list_code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-triton-operators",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert list_code == 0
    captured = capsys.readouterr()
    list_out = captured.out + captured.err
    list_kernel_names = set(re.findall(r"([^\s]+) \(id \d+\)", list_out))
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--triton-operator",
        "*",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Triton operator filter selected" in out
    kernel_top = pd.read_csv(
        Path(ml_api_trace_profiled_workload) / "pmc_kernel_top.csv"
    )
    top_names = {simplify_kernel_name(str(name)) for name in kernel_top["Kernel_Name"]}
    assert top_names <= list_kernel_names
    assert not any("aten::" in str(name) for name in kernel_top["Kernel_Name"])


def test_filter_torch_confines_metric_kernel_ids(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    list_code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert list_code == 0
    captured = capsys.readouterr()
    list_out = captured.out + captured.err
    list_kernel_names = set(re.findall(r"([^\s]+) \(id \d+\)", list_out))
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--torch-operator",
        "*",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "PyTorch operator filter selected" in out
    kernel_top = pd.read_csv(
        Path(ml_api_trace_profiled_workload) / "pmc_kernel_top.csv"
    )
    top_names = {simplify_kernel_name(str(name)) for name in kernel_top["Kernel_Name"]}
    assert top_names <= list_kernel_names
    assert not any(
        "triton.JITFunction" in str(name) for name in kernel_top["Kernel_Name"]
    )


def test_filter_torch_and_triton_together(
    ml_api_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--torch-operator",
        "*",
        "--triton-operator",
        "*",
        "--path",
        ml_api_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Matched PyTorch, Triton Operators:" in out
