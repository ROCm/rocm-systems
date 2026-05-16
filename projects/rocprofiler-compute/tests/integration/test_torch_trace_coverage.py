# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX marker coverage test for inject_roctx.py.

Samples ATen operators and structural patterns (nn.Module forward,
Optimizer.step, autograd/compile/jit/distributed/cuda surfaces), generates
a workload + a ground-truth runner that profiles each op with
torch.profiler, runs rocprof-compute --torch-trace on the workload, then
compares ROCTX markers and kernel correlations against the ground truth.

Sampling controlled by --coverage-seed / --coverage-n (see conftest.py).
On ground-truth subprocess failure, the generated workload and runner are
copied to the pytest cwd as failed_torch_trace_coverage_{workload,runner}.py.

Torch-dependent helpers live in torch_trace_coverage_utils and are
imported lazily inside the test body, after require_torch_gpu has run.
"""

import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import common

COVERAGE_TEST_CONFIG: Dict[str, Any] = {"cleanup": True}


# -- Main test --


@pytest.fixture
def torch_trace_coverage_sampling(request):
    """Return (seed, sample_budget) for test_random_operator_kernel_coverage."""
    seed = request.config.getoption("--coverage-seed")
    n = request.config.getoption("--coverage-n")
    if n < 0:
        pytest.fail("--coverage-n must be non-negative")
    return seed, n


@pytest.mark.torch_trace
def test_random_operator_kernel_coverage(
    require_torch_gpu,
    request,
    binary_handler_profile_rocprof_compute,
    torch_trace_coverage_sampling,
):
    """Verify --torch-trace ROCTX output matches profiler ground truth.

    Steps: sample ops → emit workload + runner → run runner for JSON → run
    rocprof-compute on the workload → parse CSVs → compare per op. Per-op
    mismatches are reported (stdout + UserWarning) but do not
    individually fail the test item; the test fails only if no sampled
    operator passes (a regression guard while coverage gaps remain).
    """
    from torch_trace_coverage_utils import (
        compare_single_op,
        discover_operators,
        multiline_coverage_failure_warning,
        parse_roctx_markers,
        print_torch_trace_coverage_session_header,
        run_ground_truth_torch_profiler_subprocess,
        unique_get_output_param_id,
        write_coverage_workload_artifacts,
    )

    seed, sample_budget = torch_trace_coverage_sampling
    rng = random.Random(seed)

    aten_ops, structural_ops = discover_operators()

    # sample_budget caps only the ATen sample; every structural entry is
    # always included. When the budget is smaller than len(structural_ops)
    # the resulting sample size equals len(structural_ops); see the
    # --coverage-n help text in conftest.py.
    n_aten = min(
        max(0, sample_budget - len(structural_ops)),
        len(aten_ops),
    )
    sampled = rng.sample(aten_ops, n_aten) + structural_ops

    print_torch_trace_coverage_session_header(
        seed,
        sample_budget,
        len(sampled),
        len(aten_ops),
        len(structural_ops),
    )

    gt_work_dir = common.get_output_dir(
        param_id=unique_get_output_param_id("torch_trace_gt"),
        suffix="_tmp",
        clean_existing=True,
    )
    workload_dir = common.get_output_dir(
        param_id=unique_get_output_param_id("random_op_coverage"),
        clean_existing=True,
    )
    Path(gt_work_dir).mkdir(parents=True, exist_ok=True)
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    ground_truth_path = str(Path(gt_work_dir) / "ground_truth.json")
    workload_script_path = str(Path(gt_work_dir) / "coverage_workload.py")
    ground_truth_runner_script_path = str(
        Path(gt_work_dir) / "coverage_ground_truth_runner.py"
    )

    try:
        write_coverage_workload_artifacts(
            sampled,
            workload_script_path,
            ground_truth_runner_script_path,
        )

        # Run 1: torch.profiler ground truth (runner loads workload module)
        run_ground_truth_torch_profiler_subprocess(
            ground_truth_runner_script_path,
            workload_script_path,
            ground_truth_path,
            coverage_seed=seed,
            coverage_sample_budget=sample_budget,
        )
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Run 2: rocprof-compute --torch-trace (profiled app is minimal workload)
        binary_handler_profile_rocprof_compute(
            {
                **COVERAGE_TEST_CONFIG,
                "coverage_workload": [
                    sys.executable,
                    workload_script_path,
                ],
            },
            workload_dir,
            ["--experimental", "--torch-trace", "--iteration-multiplexing"],
            check_success=False,
            app_name="coverage_workload",
        )

        roctx_kernels_map, roctx_marker_names = parse_roctx_markers(workload_dir)

        # Per-operator comparison
        failure_detail: List[Tuple[str, str]] = []
        passed = skipped = 0
        for op in sampled:
            outcome = compare_single_op(
                op,
                ground_truth,
                roctx_marker_names,
                roctx_kernels_map,
            )
            for line in outcome.log_lines:
                print(line)
            if outcome.status == "pass":
                passed += 1
            elif outcome.status == "fail":
                failure_detail.append((op.name, outcome.reason))
            else:
                skipped += 1

        print(
            f"\n  Summary: {len(sampled)} ops — "
            f"{passed} PASS, {len(failure_detail)} FAIL, {skipped} SKIP\n"
        )

        # TODO: tighten to assert not failure_detail once every sampled
        # operator reliably matches a ROCTX marker and its kernels. The
        # current assertion guards only against total regression
        # (zero successes).
        if failure_detail:
            warnings.warn(
                multiline_coverage_failure_warning(
                    failure_detail,
                    max_ops=48,
                    seed=seed,
                    sample_budget=sample_budget,
                ),
                UserWarning,
                stacklevel=1,
            )
        assert passed > 0, (
            f"no operators PASSed ROCTX/kernel coverage "
            f"(sampled={len(sampled)}, FAIL={len(failure_detail)}, SKIP={skipped})"
        )
    finally:
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            workload_dir,
        )
        common.clean_output_dir(
            COVERAGE_TEST_CONFIG["cleanup"],
            gt_work_dir,
        )

# ---------------------------------------------------------------------------
# torch_trace tests merged from tests/test_profile_general.py during the
# test-suite refactor. The test functions below are extracted verbatim; only
# the surrounding module-level imports were added.
# ---------------------------------------------------------------------------

import csv
import os
import re
import time
from pathlib import Path

import common
import pandas as pd

from _profile_helpers import config, num_devices, num_kernels

@pytest.mark.torch_trace
def test_torch_trace_profile(
    require_torch_gpu,
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """
    Test profile and analyze flow for PyTorch torch-trace.

    Runs profiling with --torch-trace, verifies profile outputs (pmc_perf, marker
    and counter CSVs), then runs analyze with --list-torch-operators and
    --torch-operator (PurePosixPath glob patterns like *relu, all), and verifies
    torch_trace directory, consolidated CSV contents (hierarchy, kernel, counters),
    and CLI output format (call tree grouped by source location, aggregated stats,
    kernel IDs, sort order).
    Requires PyTorch and GPU; not included in default suite.
    """
    workload_dir = common.get_output_dir(param_id="torch_trace")

    # --torch-trace needs --experimental for profiling
    options = [
        "--experimental",
        "--torch-trace",
        "--iteration-multiplexing",
    ]

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        app_name="torch_test_app",
    )

    # ---- Verify profiling output (checks 1–5) ----

    # 1. Profiling completed successfully
    assert returncode == 0, "Profiling the torch application failed"

    # 2. Validate profile outputs (PMC data validated by check_csv_files)
    num_devices = config.get("num_devices", 1)
    common.check_csv_files(workload_dir, num_devices, 1)

    # 3. Marker/counter CSV pairs exist and counts match
    marker_api_trace_files = list(Path(workload_dir).glob("**/*marker_api_trace.csv"))
    counter_collection_files = list(
        Path(workload_dir).glob("**/*counter_collection.csv")
    )
    assert len(marker_api_trace_files) == len(counter_collection_files), (
        "Mismatch in number of marker_api_trace.csv and counter_collection.csv files"
    )
    for marker_file in marker_api_trace_files:
        corresponding_counter_file = marker_file.parent / marker_file.name.replace(
            "marker_api_trace", "counter_collection"
        )
        assert corresponding_counter_file.exists(), (
            f"counter_collection.csv not found for {marker_file}"
        )
        # 4. marker_api_trace CSVs: required columns and non-empty rows
        expected_marker_columns = {
            "Domain",
            "Function",
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        }
        with open(marker_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {marker_file}"
            for column in expected_marker_columns:
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
        # 5. counter_collection CSVs: required columns and non-empty rows
        expected_counter_columns = {
            "Correlation_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp",
            "End_Timestamp",
        }
        with open(corresponding_counter_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames is not None, f"No columns in {corresponding_counter_file}"
            for column in expected_counter_columns:
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

    # Flush any profiling output so capsys captures only the analyze output
    capsys.readouterr()

    # ---- Verify analysis output from --list-torch-operators (checks 6–8) ----

    # 6. Analyze with --list-torch-operators succeeds
    returncode_analyze = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-torch-operators",
    ])
    assert returncode_analyze == 0, "Analyze with --list-torch-operators failed"

    list_output = capsys.readouterr().out

    # 7. torch_trace directory created with consolidated.csv
    torch_trace_dir = Path(workload_dir) / "torch_trace"
    assert torch_trace_dir.exists(), "torch_trace directory not created"

    consolidated_csv = torch_trace_dir / "consolidated.csv"
    assert consolidated_csv.exists(), "consolidated.csv not found in torch_trace"

    # 8. Consolidated CSV contains hierarchy, kernel names, and counter values
    df = pd.read_csv(consolidated_csv)
    assert not df.empty, "consolidated.csv is empty"
    assert "Operator_Name" in df.columns, "Operator_Name column missing"
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

    # ---- Verify --list-torch-operators CLI output format (checks 9–14) ----

    # 9. Banner
    assert "PyTorch Operator Call Tree:" in list_output, "Missing banner line"

    # 10. Source-location grouping (file:line headers)
    location_headers = re.findall(
        r"^(\S+:\d+)\s+\(dispatches:", list_output, re.MULTILINE
    )
    assert location_headers, "No source-location headers found in output"

    # 11. Aggregated stats on tree nodes
    assert re.search(r"\(dispatches:\s+\d+,\s+total:", list_output), (
        "No aggregated stats found in output"
    )

    # 12. Kernel IDs
    kernel_ids = re.findall(r"\(id (\d+)\)", list_output)
    assert kernel_ids, "No kernel IDs found in output"

    # 13. Kernel launch durations
    assert re.search(r"dispatches:\s+\d+,\s+total:", list_output), (
        "No kernel duration info in output"
    )

    # 14. Source locations sorted by descending total duration
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

    # 15. --list-torch-operators succeeds at every --kernel-verbose level 0-4
    #     (level 5 is the baseline run above)
    for verbose_level in range(5):
        capsys.readouterr()
        rc = binary_handler_analyze_rocprof_compute([
            "--experimental",
            "analyze",
            "--path",
            workload_dir,
            "--list-torch-operators",
            "--kernel-verbose",
            str(verbose_level),
        ])
        assert rc == 0, (
            f"--list-torch-operators failed with --kernel-verbose {verbose_level}"
        )

    # ---- Verify analysis output from --torch-operator (check 16) ----

    # Analyze with --torch-operator needs --experimental flag
    returncode_analyze_relu = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
    ])
    # 16. Analyze with --torch-operator *relu succeeds
    assert returncode_analyze_relu == 0, "Analyze with --torch-operator *relu failed"

    # --- Verify torch-operator cli output ---

    # 17. Multi-component pattern matches operator in the middle of hierarchy.
    #     torch.nn.functional.relu is a wrapper that delegates to torch.relu;
    #     only the leaf operator appears in consolidated trace, so we use
    #     wildcards to match through the hierarchy.
    capsys.readouterr()
    rc_exact = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*/torch.nn.functional.relu/*",
    ])
    assert rc_exact == 0, (
        "Analyze with --torch-operator */torch.nn.functional.relu/* failed"
    )
    out_exact = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_exact, (
        "Expected 'Matched PyTorch Operators' header in --torch-operator output"
    )
    assert "dispatches" in out_exact, (
        "Expected call tree with dispatches stats in --torch-operator output"
    )

    # 18. Glob wildcard pattern (*relu) matches the relu operator
    capsys.readouterr()
    rc_glob = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
    ])
    assert rc_glob == 0, "Analyze with --torch-operator *relu failed"
    out_glob = capsys.readouterr().out
    assert "dispatches" in out_glob, (
        "Glob pattern *relu should match relu operator and render call tree"
    )

    # 19. 'all' keyword matches every operator
    capsys.readouterr()
    rc_all = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "all",
    ])
    assert rc_all == 0, "Analyze with --torch-operator all failed"
    out_all = capsys.readouterr().out
    assert "dispatches" in out_all, "'all' keyword should match operators"

    # 20. --torch-operator + -k intersection succeeds and renders call tree
    capsys.readouterr()
    rc_intersect = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "all",
        "-k",
        "0",
    ])
    assert rc_intersect == 0, "Analyze with --torch-operator all -k 0 failed"
    out_intersect = capsys.readouterr().out
    assert "Matched PyTorch Operators" in out_intersect, (
        "Expected call tree output with --torch-operator all -k 0"
    )
    assert "Torch operator filter selected" in out_intersect, (
        "Expected filter-selection log confirming -k intersection"
    )

    # 21. Non-matching pattern degrades gracefully with a warning
    capsys.readouterr()
    rc_nomatch = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "nonexistent_operator_xyz",
    ])
    assert rc_nomatch == 0, (
        "Analyze with non-matching --torch-operator should not crash"
    )
    out_nomatch = capsys.readouterr().out
    assert "No operators matched" in out_nomatch, (
        "Expected warning about no operators matched"
    )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.torch_trace
def test_torch_trace_overhead(
    require_torch_gpu, binary_handler_profile_rocprof_compute
):
    """
    Measure overhead introduced by --torch-trace flag.
    Compares execution time with and without the flag to ensure overhead is acceptable.
    NOTE: Not included in the test suite since this requires PyTorch and GPU.
    """
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

    # Read baseline timestamps
    baseline_results_files = list(Path(workload_dir_baseline).glob("results_*.csv"))
    baseline_df = pd.concat(
        [pd.read_csv(f) for f in baseline_results_files], ignore_index=True
    )
    baseline_kernel_duration_total = (
        baseline_df["End_Timestamp"].max() - baseline_df["Start_Timestamp"].min()
    )
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
    # Read with-flag timestamps
    with_flag_results_files = list(Path(workload_dir_with_flag).glob("results_*.csv"))
    with_flag_df = pd.concat(
        [pd.read_csv(f) for f in with_flag_results_files], ignore_index=True
    )
    with_flag_kernel_duration_total = (
        with_flag_df["End_Timestamp"].max() - with_flag_df["Start_Timestamp"].min()
    )
    longest_running_kernel_baseline = (
        baseline_df["End_Timestamp"] - baseline_df["Start_Timestamp"]
    ).max()
    longest_running_kernel_with_flag = (
        with_flag_df["End_Timestamp"] - with_flag_df["Start_Timestamp"]
    ).max()
    # Calculate overheads
    longest_running_kernel_overhead = (
        (longest_running_kernel_with_flag - longest_running_kernel_baseline)
        / longest_running_kernel_baseline
    ) * 100
    wall_clock_overhead = ((with_flag_time - baseline_time) / baseline_time) * 100
    kernel_overhead = (
        (with_flag_kernel_duration_total - baseline_kernel_duration_total)
        / baseline_kernel_duration_total
    ) * 100
    print(f"\n{'=' * 70}")
    print("Performance Overhead Analysis:")
    print(f"  Longest running kernel overhead: {longest_running_kernel_overhead:.1f}%")
    print(f"  Baseline wall-clock time:     {baseline_time:.2f}s")
    print(f"  With --torch-trace time:  {with_flag_time:.2f}s")
    print(f"  Wall-clock overhead:          {wall_clock_overhead:.1f}%")
    print(f"  Baseline kernel duration:     {baseline_kernel_duration_total:.0f} ns")
    print(f"  With flag kernel duration:    {with_flag_kernel_duration_total:.0f} ns")
    print(f"  Kernel execution overhead:    {kernel_overhead:.1f}%")
    print(f"{'=' * 70}\n")

    common.clean_output_dir(config["cleanup"], workload_dir_with_flag)
    # Assert overhead is reasonable (< 100% wall-clock, < 50% kernel)
    assert wall_clock_overhead < 100, (
        f"Wall-clock overhead too high: {wall_clock_overhead:.1f}%"
    )
    assert kernel_overhead < 50, (
        f"Kernel execution overhead too high: {kernel_overhead:.1f}%"
    )
    assert longest_running_kernel_overhead < 50, (
        f"longest running kernel increase too high: "
        f"{longest_running_kernel_overhead:.1f}%"
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
    require_torch_gpu,
    binary_handler_profile_rocprof_compute,
    workload_cmd,
    expected_exit,
    request,
):
    """Integration test: workload validation exit codes with --torch-trace."""
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
