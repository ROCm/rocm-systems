# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import os
import shutil
import tempfile
from pathlib import Path

import common
import pandas as pd
import pytest

from pc_sampling.per_kernel_isa_export import PER_KERNEL_DIRECTORY_NAME
from tests.integration import common as integration_common

config = {}
config["cleanup"] = True

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
    "tests/workloads/vcopy/MI350",
    "tests/workloads/vcopy/RDNA35_HALO",
]

PC_SAMPLING_SOURCE_WORKLOAD = (
    Path(__file__).resolve().parents[1]
    / "workloads"
    / "vcopy_pc_sampling_only"
    / "MI350"
)
SOURCE_WORKLOAD_NAME = "pc_sampling_source_workload"
SOURCE_WORKLOAD_SUB_NAME = "run_001"


def setup_pc_sampling_source_workload(tmp_path):
    """Copy the PC-sampling source fixture into a temporary workload."""
    workload_path = tmp_path / SOURCE_WORKLOAD_NAME / SOURCE_WORKLOAD_SUB_NAME
    shutil.copytree(PC_SAMPLING_SOURCE_WORKLOAD, workload_path)
    return workload_path


@pytest.mark.misc
def test_valid_path(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_list_kernels(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-stats",
        ])
        assert code == 0
        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx90a(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--list-metrics",
        "gfx90a",
    ])
    assert code == 0

    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-metrics",
            "gfx90a",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--list-metrics",
        "gfx908",
    ])
    assert code == 0

    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-metrics",
            "gfx908",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908_with_block(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--list-metrics",
        "gfx908",
        "--block",
        "1",
    ])
    assert code == 1

    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-metrics",
            "gfx908",
            "--block",
            "1",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_available_metrics(binary_handler_analyze_rocprof_compute, capsys):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--list-available-metrics",
    ])
    assert code == 1

    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        try:
            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--path",
                workload_dir,
                "--list-available-metrics",
            ])
            assert code == 0

            # Test output
            output = capsys.readouterr().out
            assert "0 -> Top Stats" in output
            assert "1 -> System Info" in output
        finally:
            common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_available_metrics_with_block(
    binary_handler_analyze_rocprof_compute, capsys
):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--list-available-metrics",
        "--block",
        "1",
    ])
    assert code == 1

    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-available-metrics",
            "--block",
            "1",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "1",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "5",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "5.2.2",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "6.1",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "10",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_6(binary_handler_analyze_rocprof_compute, capsys):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "100",
        ])
        captured = capsys.readouterr()
        error_output = captured.err + captured.out
        assert code != 0
        assert "Invalid --block value '100'" in error_output

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "1",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "0",
            "1",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "1",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "2",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "3",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "2",
            "5",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "6",
            "7",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        gpu_id = "0"
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--gpu-id",
            gpu_id,
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_wave(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--normal-unit",
            "per_wave",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_cycle(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--normal-unit",
            "per_cycle",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_second(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--normal-unit",
            "per_second",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_kernel(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--normal-unit",
            "per_kernel",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--max-stat-num",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--max-stat-num",
            "5",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--max-stat-num",
            "10",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--max-stat-num",
            "15",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_s(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--time-unit",
            "s",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ms(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--time-unit",
            "ms",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_us(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--time-unit",
            "us",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ns(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--time-unit",
            "ns",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--decimal",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--decimal",
            "1",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--decimal",
            "4",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--cols",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--cols",
            "2",
            "--include-cols",
            "Description",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--cols",
            "0",
            "2",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_g(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "-g",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_baseline(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        "tests/workloads/vcopy/MI200",
        "--path",
        "tests/workloads/vcopy/MI100",
    ])
    assert code == 0

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        "tests/workloads/vcopy/MI200",
        "--path",
        "tests/workloads/vcopy/MI200",
    ])
    assert code == 1

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        "tests/workloads/vcopy/MI100",
        "--path",
        "tests/workloads/vcopy/MI100",
    ])
    assert code == 1


# =============================================================================
# Test cases for Parser.py
# =============================================================================


@pytest.mark.misc
def test_dependency_MI100(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = integration_common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dependency",
        ])
        assert code == 0
    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_missing_file_handling(binary_handler_analyze_rocprof_compute):
    """Test handling of missing files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        code = binary_handler_analyze_rocprof_compute(["analyze", "--path", temp_dir])
        assert code != 0


@pytest.mark.misc
def test_filter_combinations_coverage(binary_handler_analyze_rocprof_compute, capsys):
    """Test basic filters that should work"""
    for dir in ["tests/workloads/vcopy/MI100", "tests/workloads/vcopy/MI200"]:
        if os.path.exists(dir):
            workload_dir = integration_common.setup_workload_dir(dir)

            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--path",
                workload_dir,
            ])
            assert code == 0

            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "SQ",
            ])
            captured = capsys.readouterr()
            error_output = captured.err + captured.out
            assert code != 0
            assert "Invalid --block value 'SQ'" in error_output

            common.clean_output_dir(config["cleanup"], workload_dir)
            break


@pytest.mark.misc
def test_missing_files_scenarios(binary_handler_analyze_rocprof_compute):
    """Test scenarios with missing files to cover error paths"""
    for dir in ["tests/workloads/vcopy/MI100", "tests/workloads/vcopy/MI200"]:
        if os.path.exists(dir):
            with tempfile.TemporaryDirectory() as temp_dir:
                workload_dir = os.path.join(temp_dir, "incomplete_workload")
                shutil.copytree(dir, workload_dir)

                csv_files = ["pmc_perf_1.csv", "pmc_perf_2.csv", "timestamps.csv"]
                for csv_file in csv_files:
                    csv_path = os.path.join(workload_dir, csv_file)
                    if os.path.exists(csv_path):
                        os.remove(csv_path)

                binary_handler_analyze_rocprof_compute([
                    "analyze",
                    "--path",
                    workload_dir,
                ])
            break


@pytest.mark.iteration_multiplexing
def test_iteration_multiplexing(binary_handler_analyze_rocprof_compute):
    workload = "tests/workloads/vcopy_iteration_multiplexing/MI350"
    workload_dir = integration_common.setup_workload_dir(workload)

    # Test with dispatch filtering
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--dispatch",
        "1",
        "--path",
        workload_dir,
    ])
    assert code == 0

    # Test without dispatch filtering
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.torch_trace
def test_list_torch_operators_no_path(binary_handler_analyze_rocprof_compute, capsys):
    """Test --list-torch-operators fails gracefully without --path"""
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--list-torch-operators",
    ])
    assert code == 1

    captured = capsys.readouterr()
    error_output = captured.err + captured.out
    assert "-p/--path" in error_output or "required" in error_output.lower()


@pytest.mark.torch_trace
def test_list_torch_operators_no_trace_data(
    binary_handler_analyze_rocprof_compute, capsys
):
    """Test graceful handling when workload was profiled with --torch-trace but
    contains no torch operator data (e.g. a non-PyTorch workload like vcopy).
    """
    workload_dir = integration_common.setup_workload_dir(indirs[0])

    # Simulate a workload profiled with --torch-trace so the sanitize guard
    # passes, but no torch marker/counter files exist (non-torch workload).
    config_path = Path(workload_dir) / "profiling_config.yaml"
    config_path.write_text("torch_trace: true\n")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-torch-operators",
    ])
    # Should show warning but exit successfully
    assert code == 0

    output = capsys.readouterr().out
    assert "PyTorch Operators in:" in output
    assert "Total: 0 operators" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
@pytest.mark.parametrize(
    ("output_format", "output_name", "exports_per_kernel_files"),
    [
        pytest.param(
            "csv",
            "pc_sampling_source_csv",
            True,
            id="csv-exports-isa-and-source",
        ),
        pytest.param(
            "db",
            "pc_sampling_source_db",
            False,
            id="db-exports-neither",
        ),
    ],
)
def test_analyze_per_kernel_export_output_format(
    binary_handler_analyze_rocprof_compute,
    monkeypatch,
    tmp_path,
    output_format,
    output_name,
    exports_per_kernel_files,
):
    """Export each kernel's ISA and its source for CSV output only."""
    workload_path = setup_pc_sampling_source_workload(tmp_path).resolve()
    output_path = (tmp_path / output_name).resolve()
    monkeypatch.chdir(tmp_path)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        str(workload_path),
        "--block",
        "21",
        "--output-format",
        output_format,
        "--output-name",
        output_name,
    ])

    assert code == 0
    workload_export_path = (
        output_path
        / PER_KERNEL_DIRECTORY_NAME
        / SOURCE_WORKLOAD_NAME
        / SOURCE_WORKLOAD_SUB_NAME
    )
    if not exports_per_kernel_files:
        assert output_path.with_suffix(".db").is_file()
        assert not (output_path / PER_KERNEL_DIRECTORY_NAME).exists()
        return

    # Every exported file sits under the absolute path the CSV records for
    # it, so the paths below are the fixture's own capture-host paths.
    assert set(common.read_binary_file_tree(workload_export_path / "source")) == {
        Path("app/projects/rocprofiler-compute/sample/vcopy.cpp"),
        Path(
            "rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel"
            "/include/hip/amd_detail/amd_hip_runtime.h"
        ),
    }

    # Each ISA folder is one kernel of kernel.csv, named by its recorded uuid.
    isa_paths = sorted(workload_export_path.rglob("isa_*.csv"))
    assert isa_paths
    kernel_frame = pd.read_csv(output_path / "kernel.csv")
    assert {isa_path.parent.name for isa_path in isa_paths} <= {
        f"kernel_{kernel_uuid}" for kernel_uuid in kernel_frame["kernel_uuid"]
    }

    with isa_paths[0].open(newline="", encoding="utf-8") as isa_file:
        header, *rows = list(csv.reader(isa_file))
    assert header[:3] == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
    ]
    assert header[-3:] == ["Source", "Code object id", "Pid"]
    assert [row[0] for row in rows] == [
        str(line_number) for line_number in range(1, len(rows) + 1)
    ]
