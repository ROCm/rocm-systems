# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set

import common
import pandas as pd
import pytest

from pc_sampling.pc_sampling_analysis import load_pc_sample_records
from tests.integration import common as integration_common
from utils.file_io import load_pc_sampling_results

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["app_conjugate_gradient"] = [
    "./tests/conjugate_gradient",
    "--processes",
    "3",
    "--kernels",
    "spmv,spmv,update",
    "--rotate-code-objects",
]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1

CG_SPMV_KERNEL_NAME = "kernel_spmv_csr"
CG_UPDATE_KERNEL_NAME = "kernel_cg_update_reduce"
CG_KERNEL_NAMES = frozenset({CG_SPMV_KERNEL_NAME, CG_UPDATE_KERNEL_NAME})
CG_MODULE_A_NAME = "cg_module_a.hsaco"
CODE_OBJECT_INFO_SUFFIX = "_code_obj_info.json"
PC_SAMPLING_RESULTS_SUFFIX = "_ps_file_results.json"


def pc_sampling_file_pids(file_names: Iterable[str], suffix: str) -> Set[int]:
    """Return unique numeric PID prefixes for files ending in ``suffix``."""
    matching_file_names = [
        file_name for file_name in file_names if file_name.endswith(suffix)
    ]
    process_identifier_prefixes = [
        file_name[: len(file_name) - len(suffix)] for file_name in matching_file_names
    ]
    assert all(
        process_identifier_prefix.isdigit()
        for process_identifier_prefix in process_identifier_prefixes
    ), f"expected numeric PID prefixes for {matching_file_names}"

    process_identifiers = [
        int(process_identifier_prefix)
        for process_identifier_prefix in process_identifier_prefixes
    ]
    assert len(process_identifiers) == len(set(process_identifiers)), (
        f"expected unique PID prefixes for {matching_file_names}"
    )
    return set(process_identifiers)


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

    code_object_process_ids = pc_sampling_file_pids(
        code_object_files,
        CODE_OBJECT_INFO_SUFFIX,
    )
    result_process_ids = pc_sampling_file_pids(
        pc_sampling_results,
        PC_SAMPLING_RESULTS_SUFFIX,
    )
    assert result_process_ids == code_object_process_ids

    dynamic_files = {*code_object_files, *pc_sampling_results}
    remaining = file_names - dynamic_files
    assert remaining == {"sysinfo.csv"}


def collect_dispatched_cg_symbols(
    tool_data: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Map dispatched CG kernel IDs to their symbol records."""
    symbols_by_kernel_id = {
        symbol["kernel_id"]: symbol for symbol in tool_data["kernel_symbols"]
    }

    dispatched_symbols = {}
    for dispatch in tool_data["buffer_records"]["kernel_dispatch"]:
        kernel_id = dispatch["dispatch_info"]["kernel_id"]
        symbol = symbols_by_kernel_id.get(kernel_id)
        if symbol is None:
            continue
        if symbol["formatted_kernel_name"] not in CG_KERNEL_NAMES:
            continue
        dispatched_symbols[kernel_id] = symbol
    return dispatched_symbols


def sampled_spmv_code_object_id(
    tool_data: Dict[str, Any], spmv_kernel_ids: Set[int]
) -> int:
    """Return the sole code-object ID sampled for dispatched SpMV kernels."""
    sample_records = load_pc_sample_records(tool_data)
    spmv_sample_records = sample_records[
        sample_records["kernel_id"].isin(spmv_kernel_ids)
    ]
    assert not spmv_sample_records.empty

    sampled_code_object_ids = {
        int(code_object_id) for code_object_id in spmv_sample_records["code_object_id"]
    }
    assert len(sampled_code_object_ids) == 1
    return next(iter(sampled_code_object_ids))


def assert_module_a_code_object(
    tool_data: Dict[str, Any],
    code_object_id: int,
) -> None:
    """Assert a tool record catalogs the target ID as CG module A."""
    matching_code_objects = [
        code_object
        for code_object in tool_data["code_objects"]
        if code_object["code_object_id"] == code_object_id
    ]
    assert matching_code_objects, (
        f"code object {code_object_id} is absent from the process catalog"
    )

    module_uri = matching_code_objects[0].get("uri")
    assert isinstance(module_uri, str)
    module_uri_without_offset = module_uri.split("#offset=", maxsplit=1)[0]
    assert Path(module_uri_without_offset).name == CG_MODULE_A_NAME


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
    capsys,
    monkeypatch,
    sampling_method,
):
    """Correlate per-process samples with rotated CG code-object IDs."""
    integration_common.require_pc_sampling_gpu(
        is_stochastic=sampling_method == "stochastic"
    )
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
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
    assert stdout.count("rounds=400") == 3
    file_dict = integration_common.check_non_pmc_files(
        workload_dir,
        num_devices,
        1,
    )
    _assert_pc_sampling_files(file_dict, expected_count=3)

    result_process_ids = pc_sampling_file_pids(
        file_dict,
        PC_SAMPLING_RESULTS_SUFFIX,
    )
    tool_data_records = load_pc_sampling_results(workload_dir)
    assert len(tool_data_records) == 3
    metadata_process_ids = {
        tool_data["metadata"]["pid"] for tool_data in tool_data_records
    }
    assert metadata_process_ids == result_process_ids

    tool_data_by_process_id = {
        tool_data["metadata"]["pid"]: tool_data for tool_data in tool_data_records
    }
    dispatched_symbols_by_process_id = {
        process_id: collect_dispatched_cg_symbols(tool_data)
        for process_id, tool_data in tool_data_by_process_id.items()
    }

    spmv_process_ids = {
        process_id
        for process_id, symbols in dispatched_symbols_by_process_id.items()
        if any(
            symbol["formatted_kernel_name"] == CG_SPMV_KERNEL_NAME
            for symbol in symbols.values()
        )
    }
    update_process_ids = {
        process_id
        for process_id, symbols in dispatched_symbols_by_process_id.items()
        if any(
            symbol["formatted_kernel_name"] == CG_UPDATE_KERNEL_NAME
            for symbol in symbols.values()
        )
    }

    assert len(spmv_process_ids) == 2
    assert len(update_process_ids) == 1
    assert spmv_process_ids.isdisjoint(update_process_ids)
    assert spmv_process_ids | update_process_ids == result_process_ids

    sampled_code_object_ids = set()
    for process_id in spmv_process_ids:
        tool_data = tool_data_by_process_id[process_id]
        spmv_symbols = {
            kernel_id: symbol
            for kernel_id, symbol in dispatched_symbols_by_process_id[
                process_id
            ].items()
            if symbol["formatted_kernel_name"] == CG_SPMV_KERNEL_NAME
        }
        assert len(spmv_symbols) == 1

        sampled_code_object_id = sampled_spmv_code_object_id(
            tool_data,
            set(spmv_symbols),
        )
        dispatched_symbol = next(iter(spmv_symbols.values()))
        assert dispatched_symbol["code_object_id"] == sampled_code_object_id
        assert_module_a_code_object(tool_data, sampled_code_object_id)
        sampled_code_object_ids.add(sampled_code_object_id)

    assert len(sampled_code_object_ids) == 2

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--kernel",
        "0",
    ])
    assert code == 0

    captured = capsys.readouterr()
    assert "21. PC Sampling" in captured.out
    assert CG_SPMV_KERNEL_NAME in captured.out
    assert CG_UPDATE_KERNEL_NAME in captured.out

    dispatch_info_path = Path(workload_dir) / "pmc_dispatch_info.csv"
    assert dispatch_info_path.exists()
    dispatch_info = pd.read_csv(dispatch_info_path)
    assert "PID" in dispatch_info.columns
    dispatch_process_ids = set(dispatch_info["PID"].astype(int))
    assert dispatch_process_ids == result_process_ids

    common.clean_output_dir(config["cleanup"], workload_dir)


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

    common.clean_output_dir(config["cleanup"], workload_dir)
