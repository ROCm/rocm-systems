# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import sqlite3
from pathlib import Path
from typing import Mapping, Set

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["app_conjugate_gradient"] = ["./tests/conjugate_gradient/conjugate_gradient"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1

CG_KERNEL_NAMES = frozenset({"kernel_spmv_csr", "kernel_cg_update_reduce"})
CODE_OBJECT_INFO_SUFFIX = "_code_obj_info.json"
PC_SAMPLING_RESULTS_SUFFIX = "_ps_file_results.json"


def pc_sampling_file_pids(file_names: Set[str], suffix: str) -> Set[int]:
    """Return the numeric PID prefixes of the files ending in ``suffix``."""
    prefixes = [
        file_name[: -len(suffix)]
        for file_name in file_names
        if file_name.endswith(suffix)
    ]
    assert all(prefix.isdigit() for prefix in prefixes), (
        f"expected numeric PID prefixes on *{suffix}, got {prefixes}"
    )
    process_ids = {int(prefix) for prefix in prefixes}
    assert len(process_ids) == len(prefixes), f"expected unique PIDs, got {prefixes}"
    return process_ids


def _assert_pc_sampling_files(
    file_dict: Mapping[str, object], expected_count: int = 1
) -> None:
    """Assert PID-prefixed PC sampling and code-object output files."""
    file_names = set(file_dict)
    code_object_files = [
        file_name
        for file_name in file_names
        if file_name.endswith(CODE_OBJECT_INFO_SUFFIX)
    ]
    assert len(code_object_files) == expected_count, (
        f"expected {expected_count} *{CODE_OBJECT_INFO_SUFFIX}, got {code_object_files}"
    )
    pc_sampling_results = [
        file_name
        for file_name in file_names
        if file_name.endswith(PC_SAMPLING_RESULTS_SUFFIX)
    ]
    assert len(pc_sampling_results) == expected_count, (
        f"expected {expected_count} *{PC_SAMPLING_RESULTS_SUFFIX}, "
        f"got {pc_sampling_results}"
    )

    assert pc_sampling_file_pids(
        file_names, PC_SAMPLING_RESULTS_SUFFIX
    ) == pc_sampling_file_pids(file_names, CODE_OBJECT_INFO_SUFFIX)

    dynamic_files = {*code_object_files, *pc_sampling_results}
    remaining = file_names - dynamic_files
    assert remaining == {"sysinfo.csv"}


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return any(
        marker in output
        for marker in (
            # rocprof-compute's own pre-flight check against the agent configs
            "is not supported on any of the agents on this system",
            # rocprofiler-sdk, when it accepts the run and then rejects the config
            "Given PC sampling configuration is not supported",
        )
    )


def _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir):
    if is_pc_sampling_not_supported(f"{stdout}\n{stderr}"):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    integration_common.require_pc_sampling_gpu(is_stochastic=True)
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize("sampling_method", ["host_trap", "stochastic"])
def test_multiprocess_pc_sampling_distinct_code_objects(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    monkeypatch,
    sampling_method,
):
    """Assert each process keeps its own code-object ID for the same kernel.

    The workload runs both kernels in two processes, in opposite order. HIP
    assigns code-object IDs on first launch, so a kernel gets a different ID in
    each process and one ID names a different kernel in each. Analyze has to key
    on (pid, code object) throughout, never on the ID alone, or it would both
    split one kernel and merge two.
    """
    integration_common.require_pc_sampling_gpu(
        is_stochastic=sampling_method == "stochastic"
    )
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--pc-sampling-method",
        sampling_method,
    ]
    workload_dir = common.get_output_dir(param_id=sampling_method)

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_conjugate_gradient",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict, expected_count=2)

    # --output-name forbids path separators, so the db lands in the cwd; run
    # from the workload dir to keep it there for clean_output_dir to remove.
    workload_path = Path(workload_dir).resolve()
    database_name = f"cg_code_objects_{sampling_method}"
    monkeypatch.chdir(workload_path)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--output-format",
        "db",
        "--output-name",
        database_name,
    ])
    assert code == 0

    database_path = workload_path / f"{database_name}.db"
    assert database_path.is_file()
    connection = sqlite3.connect(str(database_path))
    try:
        samples = pd.read_sql_query(
            "SELECT DISTINCT pid, code_object_id, kernel_name "
            "FROM compute_pc_sampling_summary_view",
            connection,
        )
    finally:
        connection.close()

    # The HIP runtime's own code object is sampled too; only the CG kernels are
    # the subject here.
    cg_samples = samples[samples["kernel_name"].isin(CG_KERNEL_NAMES)]
    assert set(cg_samples["kernel_name"]) == CG_KERNEL_NAMES

    per_kernel = cg_samples.groupby("kernel_name").agg(
        processes=("pid", "nunique"), code_objects=("code_object_id", "nunique")
    )
    assert (per_kernel["processes"] == 2).all()
    assert (per_kernel["code_objects"] == 2).all()

    # The reverse hazard: each ID names a different kernel in the two processes.
    assert (cg_samples.groupby("code_object_id")["kernel_name"].nunique() == 2).all()

    common.clean_output_dir(config["cleanup"], str(workload_path))


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        stream=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" not in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_warning_pc_sampling_with_counters(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a multi-rank warning is printed when running with --block 21
    and another block (PC sampling with counters mode requires multiple passes)
    with multi-rank.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        stream=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_profile_then_analyze(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    End-to-end: profile with PC sampling (host_trap), then
    run analysis on the profiling output.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    workload_path = Path(workload_dir)

    kernel_top_csv = workload_path / "pmc_kernel_top.csv"
    assert kernel_top_csv.exists()
    kernel_top_header = kernel_top_csv.read_text().splitlines()[0]
    assert "Kernel_Name" in kernel_top_header
    assert "Count" in kernel_top_header
    assert "Percent" in kernel_top_header

    dispatch_info_csv = workload_path / "pmc_dispatch_info.csv"
    assert dispatch_info_csv.exists()
    dispatch_info_header = dispatch_info_csv.read_text().splitlines()[0]
    assert "Dispatch_ID" in dispatch_info_header
    assert "Kernel_Name" in dispatch_info_header
    assert "GPU_ID" in dispatch_info_header

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out
    assert "21. PC Sampling" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_with_sol_block(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    PC sampling with counter collection (--block 21 2): profiling produces the
    expected artifacts and analyze renders both counter and PC sampling panels.
    """
    integration_common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    assert common.check_file_pattern("- '21'", f"{workload_dir}/profiling_config.yaml")
    assert common.check_file_pattern("- '2'", f"{workload_dir}/profiling_config.yaml")

    # Analyze with a single kernel so the detailed PC sampling table renders.
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "2.1 System Speed-of-Light" in captured.out
    assert "21. PC Sampling" in captured.out
    # The "instruction" column header only renders when the table has rows.
    assert "instruction" in captured.out
    assert "instruction_type" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)
