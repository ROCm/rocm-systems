# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for PyTorch operator tracing during profiling."""

import csv
import re
import sys
import time
from pathlib import Path

import common
import pandas as pd
import pytest

from ml_api_tracing.marker_format import parse_marker_function
from tests.integration import common as integration_common
from tests.integration.common import (
    config,
    require_torch,
)
from utils import csv_compression

MARKER_API_COLUMNS = {
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
}
COUNTER_COLLECTION_COLUMNS = {
    "Correlation_Id",
    "Kernel_Name",
    "Counter_Name",
    "Counter_Value",
    "Start_Timestamp",
    "End_Timestamp",
}
# Caps for test_torch_trace_overhead. --torch-trace cost is host-side
# RecordFunction/ROCTX. Wall-clock of the profile job is a wide sanity
# check, not a measurement. GPU idle and kernel duration are not gated:
# they read deprecated results_*.csv.gz. A real overhead number needs a
# larger workload with repeats, not this sample.
TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT = 50.0


ADDMM_MARKER = re.compile(r"aten::addmm(?![A-Za-z0-9_])")


def check_torch_trace_markers(marker_rows):
    """Check the native collector's argument, scope, and launcher metadata."""
    backward_launchers = set()
    torch_markers = []
    for row in marker_rows:
        marker = parse_marker_function(row["Function"])
        if marker["Backend"] != "torch":
            continue
        if marker["Operator_Name"] == "torch.Tensor.backward":
            backward_launchers.add((row["Process_Id"], int(row["Thread_Id"])))
        torch_markers.append((row, marker))

    assert backward_launchers, "Missing torch.Tensor.backward wrapper markers"
    assert any(
        marker["Operator_Name"].startswith("aten::") for _, marker in torch_markers
    ), "No aten:: markers: the torch trace collector did not load"

    relu_markers = [
        (row, marker)
        for row, marker in torch_markers
        if marker["Operator_Name"] == "aten::relu"
    ]
    assert relu_markers, "Missing native aten::relu markers"
    for row, marker in relu_markers:
        assert "|scope=FUNCTION|" in row["Function"], "Incorrect aten::relu scope"
        assert marker["args"] == "(self=float32[5x20])", (
            f"Incorrect aten::relu tensor metadata: {marker['args']}"
        )

    backward_markers = [
        (row, marker)
        for row, marker in torch_markers
        if "|scope=BACKWARD_FUNCTION|" in row["Function"]
    ]
    assert backward_markers, "Missing native BACKWARD_FUNCTION markers"
    for row, marker in backward_markers:
        launcher_tid = marker["launcher_thread_id"]
        assert launcher_tid.isdecimal(), f"Invalid backward launcher ID: {launcher_tid}"
        assert (row["Process_Id"], int(launcher_tid)) in backward_launchers, (
            f"Backward launcher ID {launcher_tid} does not match a Python backward "
            f"thread in process {row['Process_Id']}"
        )


def percent_overhead(with_flag, baseline, label):
    """Return ``(with_flag - baseline) / baseline * 100``, or fail if baseline is 0."""
    if baseline <= 0.0:
        pytest.fail("baseline %s is %s; cannot compute overhead" % (label, baseline))
    return ((with_flag - baseline) / baseline) * 100.0


def format_duration(seconds):
    """Format a duration in seconds for overhead-test logs (s / ms / us / ns)."""
    ns = float(seconds) * 1e9
    abs_ns = abs(ns)
    if abs_ns >= 1e9:
        return f"{ns / 1e9:.3f} s"
    if abs_ns >= 1e6:
        return f"{ns / 1e6:.3f} ms"
    if abs_ns >= 1e3:
        return f"{ns / 1e3:.3f} us"
    return f"{ns:.0f} ns"


def print_torch_trace_overhead_report(wall_clock):
    """Print without/with/overhead for ``test_torch_trace_overhead``.

    ``wall_clock`` is ``(without, with_flag, overhead_pct)`` in seconds.
    """
    without, with_flag, overhead_pct = wall_clock
    print(f"\n{'=' * 72}")
    print("--torch-trace overhead")
    print(f"  {'metric':<22} {'without':>12}  {'with':>16}  {'overhead':>10}")
    print(f"  {'-' * 22} {'-' * 12}  {'-' * 16}  {'-' * 10}")
    print(
        f"  {'wall-clock':<22} {format_duration(without):>12}  "
        f"{format_duration(with_flag):>16}"
        f"  {f'{overhead_pct:+.1f}%':>10}"
    )
    print(f"{'=' * 72}\n")


@pytest.fixture(scope="module")
def torch_trace_workload_state():
    """Clean the shared profiled workload directory at module teardown.

    ``dir`` is set as soon as the directory exists so teardown always cleans it.
    ``profiled`` gates reuse, so a failed profile is not silently handed to the
    tests that follow.
    """
    state = {"dir": None, "profiled": False}
    yield state
    if state["dir"] is not None:
        common.clean_output_dir(config["cleanup"], state["dir"])


@pytest.fixture
def torch_trace_profiled_workload(
    torch_trace_workload_state,
    binary_handler_profile_rocprof_compute,
):
    """Profile simple_net with --torch-trace and return the workload directory."""
    require_torch(gpu=True)
    if not torch_trace_workload_state["profiled"]:
        workload_dir = common.get_output_dir(param_id="torch_trace")
        torch_trace_workload_state["dir"] = workload_dir
        # Run the workload with the same interpreter as pytest.
        profile_config = dict(config)
        profile_config["torch_test_app"] = [
            sys.executable,
            *config["torch_test_app"][1:],
        ]
        returncode = binary_handler_profile_rocprof_compute(
            profile_config,
            workload_dir,
            [
                "--experimental",
                "--torch-trace",
                "--iteration-multiplexing",
            ],
            check_success=True,
            app_name="torch_test_app",
        )
        assert returncode == 0, "Profiling the torch application failed"
        torch_trace_workload_state["profiled"] = True
    return torch_trace_workload_state["dir"]


def test_torch_trace_profile_csvs(torch_trace_profiled_workload):
    """Check profile CSVs and Torch argument, scope, and launcher metadata."""
    workload_dir = torch_trace_profiled_workload
    integration_common.check_csv_files(workload_dir, config.get("num_devices", 1), 1)

    marker_api_trace_files = list(
        Path(workload_dir).glob("**/*marker_api_trace.csv.gz")
    )
    assert marker_api_trace_files, "No marker_api_trace.csv.gz produced"
    for marker_file in marker_api_trace_files:
        corresponding_counter_file = marker_file.parent / marker_file.name.replace(
            "marker_api_trace", "counter_collection"
        )
        assert corresponding_counter_file.is_file(), (
            f"counter_collection CSV not found for {marker_file}"
        )
        with csv_compression.open_gzip_csv_read(marker_file) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {marker_file}"
            for column in MARKER_API_COLUMNS:
                assert column in fieldnames, (
                    f"Column '{column}' missing in {marker_file}"
                )
            marker_rows = list(reader)
            assert marker_rows, f"{marker_file} is empty"
            functions = []
            for row in marker_rows:
                functions.append(row["Function"])
                assert row["Function"], f"Empty Function in {marker_file}"
                assert row["Correlation_Id"], f"Empty Correlation ID in {marker_file}"
                assert row["Start_Timestamp"], f"Empty Start_Timestamp in {marker_file}"
                assert row["End_Timestamp"], f"Empty End_Timestamp in {marker_file}"
            assert any("nn.Module.Linear.forward" in fn for fn in functions)
            assert any(ADDMM_MARKER.search(fn) for fn in functions)
            assert any("scope=FUNCTION" in fn for fn in functions)
            assert any("|seqNr=" in fn for fn in functions)
            assert any("|tid=" in fn for fn in functions)
            assert any("|ftid=" in fn for fn in functions)
            assert any("|ltid=" in fn for fn in functions)
            assert any("|scope=" in fn for fn in functions)
            assert any("|args=" in fn for fn in functions)
            assert any(fn.endswith("|torch") for fn in functions)
        check_torch_trace_markers(marker_rows)
        with csv_compression.open_gzip_csv_read(corresponding_counter_file) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {corresponding_counter_file}"
            for column in COUNTER_COLLECTION_COLUMNS:
                assert column in fieldnames, (
                    f"Column '{column}' missing in {corresponding_counter_file}"
                )
            found_row = False
            for row in reader:
                found_row = True
                assert row["Correlation_Id"], (
                    f"Empty Correlation_Id in {corresponding_counter_file}"
                )
                assert row["Kernel_Name"], (
                    f"Empty Kernel_Name in {corresponding_counter_file}"
                )
                assert row["Counter_Name"], (
                    f"Empty Counter_Name in {corresponding_counter_file}"
                )
                assert row["Start_Timestamp"], (
                    f"Empty Start_Timestamp in {corresponding_counter_file}"
                )
                assert row["End_Timestamp"], (
                    f"Empty End_Timestamp in {corresponding_counter_file}"
                )
            assert found_row, f"{corresponding_counter_file} is empty"


def test_torch_trace_overhead(binary_handler_profile_rocprof_compute):
    """Compare host wall-clock with and without --torch-trace.

    Torch-trace adds host-side RecordFunction/ROCTX work. Asserts a wide
    wall-clock sanity cap. GPU idle and kernel duration are not gated:
    they depended on deprecated results_*.csv.gz.
    """
    require_torch(gpu=True)
    profile_config = dict(config)
    profile_config["torch_test_app"] = [
        sys.executable,
        *config["torch_test_app"][1:],
    ]
    # Run WITHOUT --torch-trace (baseline)
    workload_dir_baseline = common.get_output_dir(param_id="torch_trace_baseline")
    start_baseline = time.time()
    returncode_baseline = binary_handler_profile_rocprof_compute(
        profile_config,
        workload_dir_baseline,
        ["--iteration-multiplexing"],  # Baseline without --torch-trace
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    baseline_time = time.time() - start_baseline
    assert returncode_baseline == 0, "Baseline profiling failed"
    common.clean_output_dir(config["cleanup"], workload_dir_baseline)

    # Run WITH --torch-trace (requires --experimental)
    workload_dir_with_flag = common.get_output_dir(param_id="torch_trace_with_flag")
    start_with_flag = time.time()
    returncode_with_flag = binary_handler_profile_rocprof_compute(
        profile_config,
        workload_dir_with_flag,
        ["--experimental", "--torch-trace", "--iteration-multiplexing"],
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    with_flag_time = time.time() - start_with_flag
    assert returncode_with_flag == 0, "Profiling with torch-trace failed"
    common.clean_output_dir(config["cleanup"], workload_dir_with_flag)

    wall_clock_overhead = percent_overhead(with_flag_time, baseline_time, "wall-clock")
    print_torch_trace_overhead_report(
        wall_clock=(baseline_time, with_flag_time, wall_clock_overhead),
    )

    assert wall_clock_overhead < TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT, (
        f"Wall-clock overhead too high: {wall_clock_overhead:.1f}% "
        f"(limit {TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT}%)"
    )


@pytest.mark.parametrize(
    "workload_cmd, expected_exit",
    [
        pytest.param(
            ["python3", "nonexistent_script_abc.py"],
            1,
            id="missing_script",
        ),
        pytest.param(
            ["python3"],
            1,
            id="bare_interpreter",
        ),
        pytest.param(
            ["python3", "-u", "-v"],
            1,
            id="flags_only",
        ),
        pytest.param(
            ["python3", "-u", "nonexistent_script_abc.py"],
            1,
            id="missing_script_after_flags",
        ),
        pytest.param(
            ["nonexistentpython3", "script.py"],
            1,
            id="nonexistent_executable",
        ),
        pytest.param(
            ["./no_such_binary"],
            1,
            id="nonexistent_binary",
        ),
    ],
)
def test_profile_invalid_workloads_torch_trace(
    binary_handler_profile_rocprof_compute,
    workload_cmd,
    expected_exit,
    request,
):
    """Assert profile exit codes for invalid workloads with --torch-trace."""
    require_torch(gpu=True)
    app_name = "test_invalid_workload"
    test_config = {**config, app_name: workload_cmd}

    workload_dir = common.get_output_dir(
        param_id=f"invalid_wl_{request.node.callspec.id}"
    )

    returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
        test_config,
        workload_dir,
        options=["--experimental", "--torch-trace", "--iteration-multiplexing"],
        check_success=False,
        app_name=app_name,
        capture_output=True,
    )

    assert returncode == expected_exit, (
        f"Expected exit code {expected_exit} for {workload_cmd}, "
        f"got {returncode}.\nstdout: {stdout}\nstderr: {stderr}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "workload_cmd, expected_exit",
    [
        pytest.param(
            ["python3", "nonexistent_script_abc.py"],
            1,
            id="missing_script",
        ),
        pytest.param(
            ["python3"],
            1,
            id="bare_interpreter",
        ),
        pytest.param(
            ["python3", "-u", "-v"],
            1,
            id="flags_only",
        ),
        pytest.param(
            ["python3", "-u", "nonexistent_script_abc.py"],
            1,
            id="missing_script_after_flags",
        ),
        pytest.param(
            ["nonexistentpython3", "script.py"],
            1,
            id="nonexistent_executable",
        ),
        pytest.param(
            ["./no_such_binary"],
            1,
            id="nonexistent_binary",
        ),
        pytest.param(
            ["python3", "-c", "print('hello')"],
            0,
            id="non_gpu_workload",
        ),
    ],
)
def test_profile_invalid_workloads_no_torch_trace(
    binary_handler_profile_rocprof_compute,
    workload_cmd,
    expected_exit,
    request,
):
    """Assert profile exit codes for invalid workloads without --torch-trace."""
    app_name = "test_invalid_workload"
    test_config = {**config, app_name: workload_cmd}

    workload_dir = common.get_output_dir(
        param_id=f"invalid_wl_{request.node.callspec.id}"
    )

    returncode, stdout, stderr = binary_handler_profile_rocprof_compute(
        test_config,
        workload_dir,
        options=[],
        check_success=False,
        app_name=app_name,
        capture_output=True,
    )

    assert returncode == expected_exit, (
        f"Expected exit code {expected_exit} for {workload_cmd}, "
        f"got {returncode}.\nstdout: {stdout}\nstderr: {stderr}"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_list_torch_operators_prints_call_tree(
    torch_trace_profiled_workload,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--path",
        torch_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "PyTorch Operator Call Tree:" in out
    assert "nn.Module.Linear.forward" in out
    assert ADDMM_MARKER.search(out)
    assert "(id " in out
    assert not (
        Path(torch_trace_profiled_workload) / "ml_api_trace" / "consolidated.csv"
    ).exists()


def test_torch_operator_addmm_selects_kernels(
    torch_trace_profiled_workload,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--torch-operator",
        "*addmm*",
        "--path",
        torch_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Matched PyTorch Operators:" in out
    assert ADDMM_MARKER.search(out)
    assert "nn.Module.Linear.forward" in out
    assert "operator filter selected" in out


def test_list_torch_operators_wins_over_filter(
    torch_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--torch-operator",
        "*addmm*",
        "--path",
        torch_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Operator filters are ignored" in out
    assert "PyTorch Operator Call Tree:" in out
    assert "Matched PyTorch Operators:" not in out


def test_torch_operator_with_dispatch_filter(
    torch_trace_profiled_workload, binary_handler_analyze_rocprof_compute, capsys
):
    list_code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
        "--path",
        torch_trace_profiled_workload,
    ])
    assert list_code == 0
    capsys.readouterr()
    dispatch_csv = Path(torch_trace_profiled_workload) / "pmc_dispatch_info.csv"
    dispatch_df = pd.read_csv(dispatch_csv)
    row_one = dispatch_df[dispatch_df["Dispatch_ID"].astype(int) == 1]
    assert not row_one.empty
    dispatch_one_kernel = str(row_one.iloc[0]["Kernel_Name"])
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--torch-operator",
        "*",
        "--dispatch",
        "1",
        "--path",
        torch_trace_profiled_workload,
    ])
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    kernel_top = pd.read_csv(Path(torch_trace_profiled_workload) / "pmc_kernel_top.csv")
    assert set(kernel_top["Kernel_Name"].astype(str)) == {dispatch_one_kernel}
    assert "operator filter selected" in out


def test_torch_trace_user_range_and_backward_thread_in_marker_csv(
    binary_handler_profile_rocprof_compute,
):
    require_torch(gpu=True)
    workload_dir = common.get_output_dir(param_id="torch_trace_user_range_backward")
    profile_config = dict(config)
    profile_config["simple_net_user_range_backward_thread"] = [
        sys.executable,
        *config["torch_test_app"][1:],
        "--user-range",
        "--backward-thread",
    ]
    try:
        returncode = binary_handler_profile_rocprof_compute(
            profile_config,
            workload_dir,
            [
                "--experimental",
                "--torch-trace",
                "--iteration-multiplexing",
            ],
            check_success=True,
            app_name="simple_net_user_range_backward_thread",
        )
        assert returncode == 0
        marker_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv.gz"))
        assert marker_files
        functions = []
        thread_ids = set()
        for marker_file in marker_files:
            with csv_compression.open_gzip_csv_read(marker_file) as f:
                for row in csv.DictReader(f):
                    functions.append(row["Function"])
                    thread_ids.add(row["Thread_Id"])
        assert any(fn == "TrainingLoop_UserDefinedMarker" for fn in functions)
        assert any("torch.Tensor.backward" in fn for fn in functions)
        assert len(thread_ids) >= 2
        assert any(re.search(r"\|ltid=\d+\|", fn) for fn in functions)
    finally:
        common.clean_output_dir(config["cleanup"], workload_dir)
