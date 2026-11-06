# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for JAX operator tracing during profiling."""

import csv
import sys
from pathlib import Path

import common
import pytest

from tests.integration.common import config, require_jax
from utils import csv_compression


def captured_analyze_output(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


def root_operator_line(out, name):
    for line in out.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("├─", "└─", "|")):
            continue
        if name in stripped:
            return line
    raise AssertionError(f"root operator {name!r} not found")


@pytest.fixture(scope="module")
def jax_trace_workload_state():
    state = {"dir": None, "profiled": False}
    yield state
    if state["dir"] is not None:
        common.clean_output_dir(config["cleanup"], state["dir"])


@pytest.fixture
def jax_trace_profiled_workload(
    jax_trace_workload_state,
    binary_handler_profile_rocprof_compute,
):
    require_jax(gpu=True)
    if not jax_trace_workload_state["profiled"]:
        workload_dir = common.get_output_dir(
            param_id=common.unique_output_param_id("jax_trace"),
        )
        jax_trace_workload_state["dir"] = workload_dir
        profile_config = dict(config)
        profile_config["jax_test_app"] = [
            sys.executable,
            *config["jax_test_app"][1:],
        ]
        returncode = binary_handler_profile_rocprof_compute(
            profile_config,
            workload_dir,
            [
                "--experimental",
                "--trace-ml-api",
                "jax",
                "--iteration-multiplexing",
            ],
            check_success=True,
            app_name="jax_test_app",
        )
        assert returncode == 0, "Profiling the jax application failed"
        jax_trace_workload_state["profiled"] = True
    return jax_trace_workload_state["dir"]


def test_jax_trace_profile_csvs(jax_trace_profiled_workload):
    workload_dir = jax_trace_profiled_workload
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
    assert any("|jax" in fn for fn in functions)
    assert any("jax.jit.train_step" in fn for fn in functions)
    assert any("jax.jit.matmul_relu" in fn for fn in functions)
    assert any("jax.dot_general" in fn for fn in functions)
    assert any("jax.pmap." in fn for fn in functions)


def test_list_jax_operators_prints_call_tree(
    jax_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-operators",
        "jax",
        "--path",
        jax_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "JAX Operator Call Tree:" in out
    assert "Operator summary" in out


def test_list_jax_operators_train_step_stats(
    jax_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-operators",
        "jax",
        "--path",
        jax_trace_profiled_workload,
    ])
    assert code == 0
    out = captured_analyze_output(capsys)
    heading = root_operator_line(out, "jax.jit.train_step")
    assert "jax_workload.py:" in heading
    assert "dispatches:" in heading
    assert "total:" in heading
    lines = out.splitlines()
    heading_index = lines.index(heading)
    kernel_children = []
    for line in lines[heading_index + 1 :]:
        stripped = line.lstrip()
        if stripped.startswith("- "):
            break
        if stripped.startswith(("├─", "└─")):
            kernel_children.append(stripped)
            continue
        if stripped and not stripped.startswith("|"):
            break
    assert kernel_children


def test_nested_jit_not_child_of_train_step(
    jax_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-operators",
        "jax",
        "--path",
        jax_trace_profiled_workload,
    ])
    assert code == 0
    out = captured_analyze_output(capsys)
    child_lines = [
        line
        for line in out.splitlines()
        if line.lstrip().startswith(("├─", "└─")) and "jax.jit.matmul_relu" in line
    ]
    assert child_lines == []


def test_direct_matmul_relu_is_root(
    jax_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-operators",
        "jax",
        "--path",
        jax_trace_profiled_workload,
    ])
    assert code == 0
    out = captured_analyze_output(capsys)
    heading = root_operator_line(out, "jax.jit.matmul_relu")
    assert "jax.jit.matmul_relu" in heading


def test_eager_dot_general_is_root(
    jax_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-operators",
        "jax",
        "--path",
        jax_trace_profiled_workload,
    ])
    assert code == 0
    out = captured_analyze_output(capsys)
    heading = root_operator_line(out, "jax.dot_general")
    assert "jax.dot_general" in heading
