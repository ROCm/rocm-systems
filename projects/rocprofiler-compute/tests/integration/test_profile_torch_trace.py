# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for PyTorch operator tracing during profiling."""

import csv
import gzip
import os
import re
import time
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    config,
    require_torch,
)

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
SIMPLE_NET_OPERATORS = ("relu", "linear", "addmm", "sum")

# Caps for test_torch_trace_overhead. Host-side RecordFunction/ROCTX should
# not inflate GPU kernel bodies; cost shows up in profile wall-clock and
# inter-kernel gaps. Measured near 0% wall on gfx950 simple_net.
_TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT = 5.0
_TORCH_TRACE_GPU_IDLE_OVERHEAD_PCT = 5.0
_TORCH_TRACE_MEAN_KERNEL_OVERHEAD_PCT = 5.0
_TORCH_TRACE_MAX_KERNEL_OVERHEAD_PCT = 5.0


def _kernel_intervals(df):
    """Return ``(start, end)`` pairs with ``end > start`` from results rows."""
    starts = df["Start_Timestamp"].astype(float)
    ends = df["End_Timestamp"].astype(float)
    return [
        (float(start), float(end))
        for start, end in zip(starts, ends)
        if end > start
    ]


def _merged_busy_and_span(intervals):
    """Return ``(union_busy, span)`` for ``intervals``.

    ``union_busy`` is time covered by at least one interval (overlaps are not
    double-counted). ``span`` is last end minus first start.
    """
    if not intervals:
        return 0.0, 0.0
    ordered = sorted(intervals, key=lambda pair: pair[0])
    span = max(end for _, end in ordered) - ordered[0][0]

    busy = 0.0
    merged_start, merged_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= merged_end:
            if end > merged_end:
                merged_end = end
            continue
        busy += merged_end - merged_start
        merged_start, merged_end = start, end
    busy += merged_end - merged_start
    return busy, span


def _gpu_idle_ns(intervals):
    """Return timeline gaps inside the kernel phase: ``span - union_busy``."""
    busy, span = _merged_busy_and_span(intervals)
    idle = span - busy
    return idle if idle > 0.0 else 0.0


def _mean_kernel_duration_ns(intervals):
    """Return the mean of ``(end - start)`` over ``intervals``."""
    if not intervals:
        return 0.0
    return sum(end - start for start, end in intervals) / float(len(intervals))


def _max_kernel_duration_ns(intervals):
    """Return the max of ``(end - start)`` over ``intervals``."""
    if not intervals:
        return 0.0
    return max(end - start for start, end in intervals)


def _percent_overhead(with_flag, baseline, label):
    """Return ``(with_flag - baseline) / baseline * 100``, or fail if baseline is 0."""
    if baseline <= 0.0:
        pytest.fail("baseline %s is %s; cannot compute overhead" % (label, baseline))
    return ((with_flag - baseline) / baseline) * 100.0


def _format_duration(seconds=None, nanoseconds=None):
    """Format a duration for overhead-test logs (s / ms / us / ns)."""
    if seconds is not None:
        ns = float(seconds) * 1e9
    elif nanoseconds is not None:
        ns = float(nanoseconds)
    else:
        raise ValueError("pass seconds= or nanoseconds=")
    abs_ns = abs(ns)
    if abs_ns >= 1e9:
        return f"{ns / 1e9:.3f} s"
    if abs_ns >= 1e6:
        return f"{ns / 1e6:.3f} ms"
    if abs_ns >= 1e3:
        return f"{ns / 1e3:.3f} us"
    return f"{ns:.0f} ns"


def _print_torch_trace_overhead_report(
    wall_clock,
    gpu_idle,
    mean_kernel,
    max_kernel,
):
    """Print without/with/overhead table for ``test_torch_trace_overhead``.

    Each argument is ``(without, with_flag, overhead_pct)``. ``wall_clock``
    values are seconds; the others are nanoseconds.
    """
    rows = [
        ("wall-clock", wall_clock, True),
        ("GPU idle (gaps)", gpu_idle, False),
        ("mean kernel duration", mean_kernel, False),
        ("max kernel duration", max_kernel, False),
    ]
    print(f"\n{'=' * 72}")
    print("--torch-trace overhead")
    print(
        f"  {'metric':<22} {'without':>12}  {'with':>16}"
        f"  {'overhead':>10}"
    )
    print(f"  {'-' * 22} {'-' * 12}  {'-' * 16}  {'-' * 10}")
    for label, (without, with_flag, overhead_pct), as_seconds in rows:
        if as_seconds:
            without_text = _format_duration(seconds=without)
            with_text = _format_duration(seconds=with_flag)
        else:
            without_text = _format_duration(nanoseconds=without)
            with_text = _format_duration(nanoseconds=with_flag)
        print(
            f"  {label:<22} {without_text:>12}  {with_text:>16}"
            f"  {f'{overhead_pct:+.1f}%':>10}"
        )
    print(f"{'=' * 72}\n")


def run_analyze(analyze_handler, workload_dir, *options):
    """Run analyze --experimental on a profiled workload directory."""
    return analyze_handler([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        *options,
    ])


def assert_operator_named(output, operator_name):
    """Assert ``operator_name`` appears in analyze output."""
    assert operator_name in output, (
        f"Expected operator {operator_name!r} in analyze output"
    )


def assert_simple_net_operator(output):
    """Assert analyze output contains relu, linear, addmm, or sum."""
    assert any(name in output for name in SIMPLE_NET_OPERATORS), (
        "Expected a SimpleNet operator name in analyze output"
    )


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
        returncode = binary_handler_profile_rocprof_compute(
            config,
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


@pytest.mark.torch_trace
def test_torch_trace_profile_csvs(torch_trace_profiled_workload):
    """Assert PMC, marker, and counter CSVs from a --torch-trace profile."""
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
        with gzip.open(marker_file, "rt", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {marker_file}"
            for column in MARKER_API_COLUMNS:
                assert column in fieldnames, (
                    f"Column '{column}' missing in {marker_file}"
                )
            found_row = False
            for row in reader:
                found_row = True
                assert row["Function"], f"Empty Function in {marker_file}"
                assert row["Correlation_Id"], f"Empty Correlation ID in {marker_file}"
                assert row["Start_Timestamp"], f"Empty Start_Timestamp in {marker_file}"
                assert row["End_Timestamp"], f"Empty End_Timestamp in {marker_file}"
            assert found_row, f"{marker_file} is empty"
        with gzip.open(
            corresponding_counter_file, "rt", newline="", encoding="utf-8"
        ) as f:
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


@pytest.mark.torch_trace
def test_list_torch_operators(
    torch_trace_profiled_workload,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """Assert --list-torch-operators call tree, relu names, and consolidated.csv.

    Repeats the listing at --kernel-verbose levels 0-4.
    """
    workload_dir = torch_trace_profiled_workload
    capsys.readouterr()

    returncode_analyze = run_analyze(
        binary_handler_analyze_rocprof_compute,
        workload_dir,
        "--list-torch-operators",
    )
    assert returncode_analyze == 0, "Analyze with --list-torch-operators failed"

    list_output = capsys.readouterr().out
    assert "PyTorch Operator Call Tree:" in list_output, "Missing banner line"
    assert_operator_named(list_output, "relu")

    location_headers = re.findall(
        r"^(\S+:\d+)\s+\(dispatches:", list_output, re.MULTILINE
    )
    assert location_headers, "No source-location headers found in output"
    assert re.search(r"\(dispatches:\s+\d+,\s+total:", list_output), (
        "No aggregated stats found in output"
    )
    kernel_ids = re.findall(r"\(id (\d+)\)", list_output)
    assert kernel_ids, "No kernel IDs found in output"

    location_durations = re.findall(
        r"^(\S+:\d+)\s+\(dispatches:\s+\d+,\s+total:\s+([\d.]+)\s+(ms|us)",
        list_output,
        re.MULTILINE,
    )
    assert location_durations, "No location durations found for sort-order check"
    durations_ms = [
        float(val) if unit == "ms" else float(val) / 1000.0
        for _, val, unit in location_durations
    ]
    assert durations_ms == sorted(durations_ms, reverse=True), (
        f"Source locations not sorted by descending duration: {location_durations}"
    )

    ml_api_trace_dir = Path(workload_dir) / "ml_api_trace"
    assert ml_api_trace_dir.exists(), "ml_api_trace directory not created"
    consolidated_csv = ml_api_trace_dir / "consolidated.csv"
    assert consolidated_csv.exists(), "consolidated.csv not found in ml_api_trace"
    df = pd.read_csv(consolidated_csv)
    assert not df.empty, "consolidated.csv is empty"
    assert "Operator_Name" in df.columns, "Operator_Name column missing"
    assert df["Operator_Name"].astype(str).str.contains("relu", case=False).any(), (
        "No relu operator in consolidated.csv"
    )
    hierarchy_present = (
        df["Operator_Name"].apply(lambda x: "/" in str(x) or "::" in str(x)).any()
    )
    assert hierarchy_present, "No hierarchy information in consolidated.csv"
    assert "Kernel_Name" in df.columns, "Kernel_Name missing"
    assert df["Kernel_Name"].notnull().all() and (df["Kernel_Name"] != "").all(), (
        "Empty Kernel_Name in consolidated.csv"
    )
    assert "Counter_Value" in df.columns, "Counter_Value column missing"
    assert df["Counter_Value"].notnull().all()
    assert (df["Counter_Value"] != "").all(), "Empty Counter_Value in consolidated.csv"

    for verbose_level in range(5):
        capsys.readouterr()
        rc = run_analyze(
            binary_handler_analyze_rocprof_compute,
            workload_dir,
            "--list-torch-operators",
            "--kernel-verbose",
            str(verbose_level),
        )
        assert rc == 0, (
            f"--list-torch-operators failed with --kernel-verbose {verbose_level}"
        )
        verbose_output = capsys.readouterr().out
        assert "PyTorch Operator Call Tree:" in verbose_output, (
            f"Missing banner at --kernel-verbose {verbose_level}"
        )
        assert_operator_named(verbose_output, "relu")


@pytest.mark.torch_trace
def test_torch_operator_filters(
    torch_trace_profiled_workload,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """Assert --torch-operator *relu*, all, -k 0, and a non-matching pattern."""
    workload_dir = torch_trace_profiled_workload
    capsys.readouterr()

    returncode_relu = run_analyze(
        binary_handler_analyze_rocprof_compute,
        workload_dir,
        "--torch-operator",
        "*relu*",
    )
    assert returncode_relu == 0, "Analyze with --torch-operator *relu* failed"
    out_relu = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_relu, (
        "Expected 'Matched PyTorch Operators' header from --torch-operator *relu*"
    )
    assert_operator_named(out_relu, "relu")

    capsys.readouterr()
    returncode_all = run_analyze(
        binary_handler_analyze_rocprof_compute,
        workload_dir,
        "--torch-operator",
        "all",
    )
    assert returncode_all == 0, "Analyze with --torch-operator all failed"
    out_all = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_all
    assert_operator_named(out_all, "relu")

    capsys.readouterr()
    returncode_intersect = run_analyze(
        binary_handler_analyze_rocprof_compute,
        workload_dir,
        "--torch-operator",
        "all",
        "-k",
        "0",
    )
    assert returncode_intersect == 0, "Analyze with --torch-operator all -k 0 failed"
    out_intersect = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_intersect, (
        "Expected call tree output with --torch-operator all -k 0"
    )
    assert "Torch operator filter selected" in out_intersect, (
        "Expected filter-selection log confirming -k intersection"
    )
    assert_simple_net_operator(out_intersect)

    capsys.readouterr()
    returncode_nomatch = run_analyze(
        binary_handler_analyze_rocprof_compute,
        workload_dir,
        "--torch-operator",
        "nonexistent_operator_xyz",
    )
    assert returncode_nomatch == 0, (
        "Analyze with non-matching --torch-operator should not crash"
    )
    out_nomatch = capsys.readouterr().out
    assert "No PyTorch operators matched" in out_nomatch, (
        "Expected warning about no operators matched"
    )


@pytest.mark.torch_trace
def test_torch_trace_overhead(binary_handler_profile_rocprof_compute):
    """Compare host and GPU timeline overhead with and without --torch-trace.

    Torch-trace adds host-side RecordFunction/ROCTX work, not slower GPU
    kernels. Asserts profile wall-clock, GPU idle gaps, and mean/max kernel
    duration.
    """
    require_torch(gpu=True)
    # Run WITHOUT --torch-trace (baseline)
    workload_dir_baseline = common.get_output_dir(param_id="torch_trace_baseline")
    start_baseline = time.time()
    returncode_baseline = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_baseline,
        ["--iteration-multiplexing"],  # Baseline without --torch-trace
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    baseline_time = time.time() - start_baseline
    assert returncode_baseline == 0, "Baseline profiling failed"

    baseline_results_files = sorted(
        Path(workload_dir_baseline).glob("results_*.csv.gz")
    )
    baseline_df = pd.concat(
        [pd.read_csv(f) for f in baseline_results_files], ignore_index=True
    )
    baseline_intervals = _kernel_intervals(baseline_df)
    baseline_idle = _gpu_idle_ns(baseline_intervals)
    _, baseline_span = _merged_busy_and_span(baseline_intervals)
    baseline_mean_kernel = _mean_kernel_duration_ns(baseline_intervals)
    baseline_max_kernel = _max_kernel_duration_ns(baseline_intervals)
    common.clean_output_dir(config["cleanup"], workload_dir_baseline)

    # Run WITH --torch-trace (requires --experimental)
    workload_dir_with_flag = common.get_output_dir(param_id="torch_trace_with_flag")
    start_with_flag = time.time()
    returncode_with_flag = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_with_flag,
        ["--experimental", "--torch-trace", "--iteration-multiplexing"],
        check_success=True,
        roof=False,
        app_name="torch_test_app",
    )
    with_flag_time = time.time() - start_with_flag
    assert returncode_with_flag == 0, "Profiling with torch-trace failed"

    with_flag_results_files = sorted(
        Path(workload_dir_with_flag).glob("results_*.csv.gz")
    )
    with_flag_df = pd.concat(
        [pd.read_csv(f) for f in with_flag_results_files], ignore_index=True
    )
    with_flag_intervals = _kernel_intervals(with_flag_df)
    with_flag_idle = _gpu_idle_ns(with_flag_intervals)
    with_flag_mean_kernel = _mean_kernel_duration_ns(with_flag_intervals)
    with_flag_max_kernel = _max_kernel_duration_ns(with_flag_intervals)

    wall_clock_overhead = _percent_overhead(
        with_flag_time, baseline_time, "wall-clock"
    )
    if baseline_idle > 0.0:
        idle_overhead = _percent_overhead(with_flag_idle, baseline_idle, "GPU idle")
    else:
        idle_growth = with_flag_idle - baseline_idle
        idle_overhead = (
            (idle_growth / baseline_span) * 100.0 if baseline_span > 0.0 else 0.0
        )
    mean_kernel_overhead = _percent_overhead(
        with_flag_mean_kernel, baseline_mean_kernel, "mean kernel duration"
    )
    max_kernel_overhead = _percent_overhead(
        with_flag_max_kernel, baseline_max_kernel, "max kernel duration"
    )

    _print_torch_trace_overhead_report(
        wall_clock=(baseline_time, with_flag_time, wall_clock_overhead),
        gpu_idle=(baseline_idle, with_flag_idle, idle_overhead),
        mean_kernel=(baseline_mean_kernel, with_flag_mean_kernel, mean_kernel_overhead),
        max_kernel=(baseline_max_kernel, with_flag_max_kernel, max_kernel_overhead),
    )

    common.clean_output_dir(config["cleanup"], workload_dir_with_flag)

    assert wall_clock_overhead < _TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT, (
        f"Wall-clock overhead too high: {wall_clock_overhead:.1f}% "
        f"(limit {_TORCH_TRACE_WALL_CLOCK_OVERHEAD_PCT}%)"
    )
    assert idle_overhead < _TORCH_TRACE_GPU_IDLE_OVERHEAD_PCT, (
        f"GPU idle (gap) overhead too high: {idle_overhead:.1f}% "
        f"(limit {_TORCH_TRACE_GPU_IDLE_OVERHEAD_PCT}%)"
    )
    assert mean_kernel_overhead < _TORCH_TRACE_MEAN_KERNEL_OVERHEAD_PCT, (
        f"Mean kernel duration overhead too high: {mean_kernel_overhead:.1f}% "
        f"(limit {_TORCH_TRACE_MEAN_KERNEL_OVERHEAD_PCT}%)"
    )
    assert max_kernel_overhead < _TORCH_TRACE_MAX_KERNEL_OVERHEAD_PCT, (
        f"Max kernel duration overhead too high: {max_kernel_overhead:.1f}% "
        f"(limit {_TORCH_TRACE_MAX_KERNEL_OVERHEAD_PCT}%)"
    )


@pytest.mark.torch_trace
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


@pytest.mark.torch_trace
def test_torch_trace_deep_tensor_wraps_overhead(
    binary_handler_profile_rocprof_compute,
):
    """Manual benchmark for the deep tensor wrap overhead.

    Skipped unless ``ROCPROFCOMPUTE_RUN_DEEP_TENSOR_WRAP_BENCH=1``.
    """
    require_torch(gpu=True)

    run_bench = os.environ.get("ROCPROFCOMPUTE_RUN_DEEP_TENSOR_WRAP_BENCH", "")
    if run_bench.strip().lower() not in ("1", "true", "yes", "on"):
        pytest.skip(
            "set ROCPROFCOMPUTE_RUN_DEEP_TENSOR_WRAP_BENCH=1 to run this benchmark"
        )

    def _run_once(*, deep_wraps: bool, param_id: str) -> tuple[float, float]:
        workload_dir = common.get_output_dir(param_id=param_id)
        prev = os.environ.get("ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS")
        os.environ["ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS"] = (
            "1" if deep_wraps else "0"
        )
        try:
            start = time.time()
            returncode = binary_handler_profile_rocprof_compute(
                config,
                workload_dir,
                ["--experimental", "--torch-trace", "--iteration-multiplexing"],
                check_success=True,
                roof=False,
                app_name="torch_test_app",
            )
            elapsed = time.time() - start
            assert returncode == 0, "torch-trace profiling run failed"

            results_files = sorted(Path(workload_dir).glob("results_*.csv.gz"))
            df = pd.concat([pd.read_csv(f) for f in results_files], ignore_index=True)
            kernel_duration_total = (
                df["End_Timestamp"].max() - df["Start_Timestamp"].min()
            )
            return elapsed, kernel_duration_total
        finally:
            common.clean_output_dir(config["cleanup"], workload_dir)
            if prev is None:
                os.environ.pop("ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS", None)
            else:
                os.environ["ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS"] = prev

    baseline_wall, baseline_kernel = _run_once(
        deep_wraps=False,
        param_id="torch_trace_deep_wraps_off",
    )
    deep_wall, deep_kernel = _run_once(
        deep_wraps=True,
        param_id="torch_trace_deep_wraps_on",
    )

    wall_overhead = (
        ((deep_wall - baseline_wall) / baseline_wall) * 100
        if baseline_wall > 0
        else 0.0
    )
    kernel_overhead = (
        ((deep_kernel - baseline_kernel) / baseline_kernel) * 100
        if baseline_kernel > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("Deep Tensor Wrap Overhead Benchmark (--torch-trace):")
    print(f"  Baseline wall-clock:         {baseline_wall:.2f}s")
    print(f"  Deep-wraps wall-clock:       {deep_wall:.2f}s")
    print(f"  Wall-clock overhead:         {wall_overhead:.1f}%")
    print(f"  Baseline kernel duration:    {baseline_kernel:.0f} ns")
    print(f"  Deep-wraps kernel duration:  {deep_kernel:.0f} ns")
    print(f"  Kernel overhead:             {kernel_overhead:.1f}%")
    print("=" * 70 + "\n")
