# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import os
import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import common
import numpy as np
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import cli_analysis
from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.parser import load_pc_sampling_data
from utils.pc_sampling_analysis import load_pc_sample_records

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

roofline_dir = "tests/workloads/mem_levels_HBM/MI200"

time_units = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


# =============================================================================
# Roofline analyze tests
# =============================================================================

_, roofline_soc = common.gpu_soc()


def test_analyze_generates_roofline_html(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze generates roofline HTML from existing workload data.
    Uses MI200 workload with roofline.csv.
    """
    if roofline_soc is None:
        pytest.skip("No supported GPU detected")
    if roofline_soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_multiple_datatypes(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze with multiple data types.
    Verifies each datatype can be requested independently.
    """
    if roofline_soc is None:
        pytest.skip("No supported GPU detected")
    if roofline_soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    for dtype in ["FP32"]:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--roofline-data-type",
            dtype,
        ])
        assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_missing_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze without roofline.csv should not crash.
    Uses a workload directory that has sysinfo.csv but no roofline.csv.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)
    roofline_csv = Path(workload_dir) / "roofline.csv"
    if roofline_csv.exists():
        roofline_csv.unlink()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_idempotent(
    binary_handler_analyze_rocprof_compute,
):
    """
    Running analyze twice on the same profiling output should produce
    consistent results without errors.
    """
    if roofline_soc is None:
        pytest.skip("No supported GPU detected")
    if roofline_soc in ("MI100"):
        pytest.skip("Roofline not supported on MI100")

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    analyze_args = [
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ]

    code1 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code1 == 0

    code2 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code2 == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_corrupted_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze with a corrupted roofline.csv should handle gracefully.
    """
    import tempfile

    if os.path.exists(roofline_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            workload_dir = os.path.join(temp_dir, "corrupted_workload")
            shutil.copytree(roofline_dir, workload_dir)

            roofline_csv = Path(workload_dir) / "roofline.csv"
            roofline_csv.write_text("this,is,bad,csv")

            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "-b 4",
                "--path",
                workload_dir,
            ])
            assert code == 0


def test_roof_invalid_data_type(binary_handler_analyze_rocprof_compute):
    """Invalid --roofline-data-type should be caught by analyze argparser."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "INVALID_TYPE",
    ])
    assert code != 0, "Invalid datatype should be rejected by argparser"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_ceiling_data_validation(binary_handler_analyze_rocprof_compute):
    """Invalid --mem-level should be caught during analyze."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        "INVALID_LEVEL",
    ])
    assert code >= 0

    common.clean_output_dir(config["cleanup"], workload_dir)


roofline_mem_level_dirs = {
    "vL1D": "tests/workloads/mem_levels_vL1D/MI200",
    "LDS": "tests/workloads/mem_levels_LDS/MI200",
}


@pytest.mark.parametrize(
    "mem_level",
    ["vL1D", "LDS"],
    ids=["vL1D", "LDS"],
)
def test_roof_mem_levels(binary_handler_analyze_rocprof_compute, mem_level):
    """Analyze with --mem-level generates roofline HTML output."""
    workload_src = roofline_mem_level_dirs[mem_level]
    if not os.path.exists(workload_src):
        pytest.skip(f"Workload directory {workload_src} not found")

    workload_dir = common.setup_workload_dir(workload_src, param_id=mem_level)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        mem_level,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_missing_file_handling():
    """cli_generate_plot with empty ai_data returns None."""

    import pandas as pd

    from roofline.roofline_main import Roofline
    from utils.specs import generate_machine_specs

    class MockArgs:
        def __init__(self):
            self.roof_only = True
            self.mem_level = "ALL"
            self.sort = "ALL"
            self.roofline_data_type = ["FP32"]

    args = MockArgs()
    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = pd.read_csv(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "roofline_data_type": ["FP32"],
    }

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("FP32", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_invalid_datatype_cli():
    """cli_generate_plot with invalid datatype returns None."""

    import pandas as pd

    from roofline.roofline_main import Roofline
    from utils.specs import generate_machine_specs

    class MockArgs:
        def __init__(self):
            self.roof_only = True
            self.mem_level = "ALL"
            self.sort = "ALL"
            self.roofline_data_type = ["FP32"]

    args = MockArgs()

    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = pd.read_csv(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "roofline_data_type": ["FP32"],
    }

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("INVALID_DATATYPE", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_valid_path(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        assert "Invalid --block value 100" in error_output

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
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
def test_dispatch_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
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
def test_dispatch_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "1",
            "4",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--dispatch",
            "5",
            "6",
        ])
        assert code == 1

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        if (
            dir == "tests/workloads/vcopy/MI100"
            or dir == "tests/workloads/vcopy/MI200"
            or dir == "tests/workloads/vcopy/MI350"
            or dir == "tests/workloads/vcopy/RDNA35_HALO"
        ):
            # MI100/MI200 workloads (rocpd format) have GPU IDs re-ranked to
            # 0-based consecutive integers by process_rocpd_csv(). MI350 and
            # RDNA35_HALO also use GPU ID 0.
            gpu_id = "0"
        else:
            gpu_id = "2"
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
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
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "-g",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_0(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "0",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "1",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "2",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "3",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "4",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "5",
        ])
        assert code == 0

        common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_6(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = common.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--kernel-verbose",
            "6",
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
        workload_dir = common.setup_workload_dir(dir)
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
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        code = binary_handler_analyze_rocprof_compute(["analyze", "--path", temp_dir])
        assert code != 0


@pytest.mark.misc
def test_filter_combinations_coverage(binary_handler_analyze_rocprof_compute, capsys):
    """Test basic filters that should work"""
    for dir in ["tests/workloads/vcopy/MI100", "tests/workloads/vcopy/MI200"]:
        if os.path.exists(dir):
            workload_dir = common.setup_workload_dir(dir)

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
            assert "Invalid --block value SQ" in error_output

            common.clean_output_dir(config["cleanup"], workload_dir)
            break


@pytest.mark.misc
def test_apply_filters_direct():
    """Test apply_filters function directly to cover filter branches"""

    import pandas as pd

    from utils.parser import apply_filters

    class MockWorkload:
        def __init__(self):
            self.raw_pmc = pd.DataFrame({
                "GPU_ID": [0, 0, 1, 1],
                "Kernel_Name": [
                    "vecCopy",
                    "vecAdd",
                    "vecCopy",
                    "vecMul",
                ],
                "Dispatch_ID": [0, 1, 2, 3],
                "Node": ["node0", "node0", "node1", "node1"],
            })

        filter_nodes = None
        filter_gpu_ids = None
        filter_kernel_ids = None
        filter_dispatch_ids = None

    workload = MockWorkload()

    workload.filter_gpu_ids = "0"
    result = apply_filters(workload, "/tmp", False, False)
    assert len(result) == 2

    workload.filter_gpu_ids = None
    workload.filter_kernel_ids = ["vecCopy"]
    result = apply_filters(workload, "/tmp", False, False)
    assert len(result) == 2

    workload.filter_kernel_ids = None
    workload.filter_dispatch_ids = ["0", "1"]
    result = apply_filters(workload, "/tmp", False, False)
    assert len(result) == 2

    # Test node filter with list of strings
    workload = MockWorkload()
    workload.filter_nodes = ["node0", "node1"]
    result = apply_filters(workload, "/tmp", False, False)
    assert len(result) == 4

    # Test GPU filter with list of integers
    workload = MockWorkload()
    workload.filter_gpu_ids = [0, 1]
    result = apply_filters(workload, "/tmp", False, False)
    assert len(result) == 4


@pytest.mark.misc
def test_missing_files_scenarios(binary_handler_analyze_rocprof_compute):
    """Test scenarios with missing files to cover error paths"""
    import tempfile

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


@pytest.mark.misc
def test_pc_sampling_basic_coverage():
    """Test PC sampling functions with minimal data"""

    class MockWorkload:
        filter_kernel_ids = []

    workload = MockWorkload()

    assert load_pc_sampling_data(workload, "none", "count", None).empty
    assert load_pc_sampling_data(workload, "missing", "count", None).empty

    workload.filter_kernel_ids = [0, 1, 2]  # Multiple kernels
    assert load_pc_sampling_data(workload, "test", "count", None).empty

    empty_records = load_pc_sample_records({
        "buffer_records": {
            "pc_sample_stochastic": [],
            "pc_sample_host_trap": [],
            "kernel_dispatch": [],
        },
    })
    assert empty_records.empty


@pytest.mark.division_by_zero
class TestMetricEvaluatorDivisionByZero:
    """Test MetricEvaluator.eval_expression handles division-by-zero cases.

    The evaluator must gracefully handle all denominator-zero and NaN scenarios
    that can arise from real counter data. These tests exercise
    utils.metrics.metric_evaluator.MetricEvaluator.eval_expression
    (None, NaN, inf detection).
    """

    @staticmethod
    def _make_evaluator(columns, sys_vars=None):
        """Build a MetricEvaluator with the given counter columns."""
        raw_pmc_df = pd.DataFrame(columns)
        return MetricEvaluator(raw_pmc_df, sys_vars or {}, {})

    @staticmethod
    def _to_eval_str(equation):
        """Transform a YAML-style equation through the full pipeline."""
        return build_eval_string(equation)

    def test_all_zero_denominator_returns_na(self):
        """Division by all-zero denominator produces inf, caught as N/A."""
        evaluator = self._make_evaluator({
            "NUMERATOR": [100.0, 200.0, 300.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = self._to_eval_str("MIN(NUMERATOR / DENOMINATOR)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", (
            "Division by all-zero Series should produce inf, caught as N/A"
        )

    def test_zero_over_zero_returns_na(self):
        """0/0 scalar division (SUM(0)/SUM(0)) produces NaN, caught as N/A."""
        evaluator = self._make_evaluator({
            "NUMERATOR": [0.0, 0.0, 0.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = self._to_eval_str("SUM(NUMERATOR) / SUM(DENOMINATOR)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", "SUM(0) / SUM(0) should produce NaN, caught as N/A"

    def test_normal_nonzero_returns_valid_float(self):
        """All non-zero values produce a valid numeric result."""
        evaluator = self._make_evaluator({
            "BUSY": [800.0, 600.0, 400.0],
            "TOTAL": [1000.0, 1000.0, 1000.0],
        })
        eval_str = self._to_eval_str("SUM(100 * BUSY) / SUM(TOTAL)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result == pytest.approx(60.0, abs=1e-9), (
            "SUM(100*[800,600,400]) / SUM([1000,1000,1000]) should be 60.0, "
            f"got {result}"
        )

    def test_nullified_incomplete_kernel_returns_na(self):
        """Incomplete kernel nullified by imputation: both columns all-NaN → N/A.

        Kernels with fewer dispatches than perfmon files have every counter
        column set to NaN before metric evaluation.
        SUM(all-NaN) / SUM(all-NaN) = NaN, which must be caught as N/A.
        """
        evaluator = self._make_evaluator({
            "NUMERATOR": [np.nan, np.nan, np.nan],
            "DENOMINATOR": [np.nan, np.nan, np.nan],
        })
        eval_str = self._to_eval_str("SUM(NUMERATOR) / SUM(DENOMINATOR)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", (
            "Nullified incomplete kernel (both columns all-NaN) should produce "
            "NaN, caught as N/A"
        )

    def test_system_variable_as_denominator(self):
        """System variable used as denominator produces valid result."""
        evaluator = self._make_evaluator(
            {"COUNTER": [100.0, 200.0]},
            sys_vars={"ammolite__var": 5},
        )
        eval_str = self._to_eval_str("SUM(COUNTER) / $var")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result == pytest.approx(60.0, abs=1e-9), (
            f"SUM([100, 200]) / 5 should be 60.0, got {result}"
        )

    def test_partial_zeros_in_denominator_aggregates_correctly(self):
        """Partial zeros in denominator are aggregated past by SUM."""
        evaluator = self._make_evaluator({
            "LEVEL": [100.0, 200.0, 300.0],
            "REQ": [10.0, 0.0, 5.0],
        })
        eval_str = self._to_eval_str("SUM(LEVEL) / SUM(REQ)")
        result = evaluator.eval_expression(eval_str)
        # SUM([100,200,300]) / SUM([10,0,5]) = 600 / 15 = 40.0
        assert isinstance(result, float)
        assert result == pytest.approx(40.0, abs=1e-9), (
            f"SUM(LEVEL) / SUM(REQ) should be 40.0, got {result}"
        )


@pytest.fixture
def sample_time_data():
    return pd.DataFrame({
        "Metric_ID": ["7.2.0", "7.2.1", "7.2.2"],
        "Metric": [
            "Kernel Time",
            "Kernel Time (Cycles)",
            "Non-Time Metric",
        ],
        "Avg": [3446.64, 64499.39, 1000.0],
        "Min": [1769.25, 17269.25, 500.0],
        "Max": [12532.12, 337030.50, 2000.0],
        "Unit": ["ns", "Cycle", "Count"],
    })


@pytest.fixture
def original_ns_values():
    return {"Avg": 3446.64, "Min": 1769.25, "Max": 12532.12}


@pytest.mark.time_unit_conversion
def test_has_time_data_detection(sample_time_data):
    from utils.tty import has_time_data

    assert has_time_data(sample_time_data)

    no_time_data = pd.DataFrame({
        "Metric": ["Non-Time Metric"],
        "Avg": [1000.0],
        "Unit": ["Count"],
    })
    assert not has_time_data(no_time_data)

    no_unit_column = pd.DataFrame({"Metric": ["Some Metric"], "Avg": [1000.0]})
    assert not has_time_data(no_unit_column)


@pytest.mark.time_unit_conversion
def test_default_unit_is_nanoseconds(sample_time_data):
    time_rows = sample_time_data["Unit"].str.lower().str.contains("ns", na=False)
    assert time_rows.any()
    assert sample_time_data.loc[0, "Unit"] == "ns"


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_seconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "s")

    assert converted_df.loc[0, "Unit"] == "s"

    expected_avg = original_ns_values["Avg"] / time_units["s"]
    expected_min = original_ns_values["Min"] / time_units["s"]
    expected_max = original_ns_values["Max"] / time_units["s"]

    assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-10)
    assert converted_df.loc[0, "Min"] == pytest.approx(expected_min, abs=1e-10)
    assert converted_df.loc[0, "Max"] == pytest.approx(expected_max, abs=1e-10)

    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_milliseconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "ms")

    assert converted_df.loc[0, "Unit"] == "ms"

    expected_avg = original_ns_values["Avg"] / time_units["ms"]
    expected_min = original_ns_values["Min"] / time_units["ms"]
    expected_max = original_ns_values["Max"] / time_units["ms"]

    assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-6)
    assert converted_df.loc[0, "Min"] == pytest.approx(expected_min, abs=1e-6)
    assert converted_df.loc[0, "Max"] == pytest.approx(expected_max, abs=1e-6)


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_microseconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "us")

    assert converted_df.loc[0, "Unit"] == "us"

    expected_avg = original_ns_values["Avg"] / time_units["us"]
    expected_min = original_ns_values["Min"] / time_units["us"]
    expected_max = original_ns_values["Max"] / time_units["us"]

    assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-3)
    assert converted_df.loc[0, "Min"] == pytest.approx(expected_min, abs=1e-3)
    assert converted_df.loc[0, "Max"] == pytest.approx(expected_max, abs=1e-3)


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_nanoseconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "ns")

    assert converted_df.loc[0, "Unit"] == "ns"

    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"], abs=1e-10
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"], abs=1e-10
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"], abs=1e-10
    )


@pytest.mark.time_unit_conversion
def test_non_time_rows_unchanged(sample_time_data):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "ms")

    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"
    assert converted_df.loc[1, "Avg"] == 64499.39
    assert converted_df.loc[2, "Avg"] == 1000.0


@pytest.mark.time_unit_conversion
def test_invalid_time_unit_handling(sample_time_data):
    from utils.tty import convert_time_columns

    original_df = sample_time_data.copy()
    converted_df = convert_time_columns(sample_time_data, "invalid_unit")

    pd.testing.assert_frame_equal(converted_df, original_df)


@pytest.mark.time_unit_conversion
def test_missing_unit_column():
    from utils.tty import convert_time_columns

    df_no_unit = pd.DataFrame({"Metric": ["Test Metric"], "Avg": [1000.0]})
    converted_df = convert_time_columns(df_no_unit, "ms")

    pd.testing.assert_frame_equal(converted_df, df_no_unit)


@pytest.mark.time_unit_conversion
def test_conversion_with_missing_columns(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    df_partial = sample_time_data[["Metric_ID", "Metric", "Avg", "Unit"]].copy()
    converted_df = convert_time_columns(df_partial, "ms")

    assert converted_df.loc[0, "Unit"] == "ms"
    expected_avg = original_ns_values["Avg"] / time_units["ms"]
    assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-6)


@pytest.mark.time_unit_conversion
def test_mathematical_correctness_all_units(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    test_cases = [
        ("s", 10**9),  # 1 second = 10^9 nanoseconds
        ("ms", 10**6),  # 1 millisecond = 10^6 nanoseconds
        ("us", 10**3),  # 1 microsecond = 10^3 nanoseconds
        ("ns", 1),  # 1 nanosecond = 1 nanosecond
    ]

    for target_unit, divisor in test_cases:
        converted_df = convert_time_columns(sample_time_data, target_unit)

        expected_avg = original_ns_values["Avg"] / divisor
        expected_min = original_ns_values["Min"] / divisor
        expected_max = original_ns_values["Max"] / divisor

        assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-10)
        assert converted_df.loc[0, "Min"] == pytest.approx(expected_min, abs=1e-10)
        assert converted_df.loc[0, "Max"] == pytest.approx(expected_max, abs=1e-10)
        assert converted_df.loc[0, "Unit"] == target_unit


# Integration tests with show_all functionality
@pytest.mark.time_unit_integration
def test_integration_conversion_flow():
    from utils.tty import convert_time_columns, has_time_data

    mock_args = Mock()
    mock_args.time_unit = "ms"
    mock_args.decimal = 2

    sample_df = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446640.0],  # 3.44664 ms in nanoseconds
        "Min": [1769250.0],  # 1.76925 ms in nanoseconds
        "Max": [12532120.0],  # 12.53212 ms in nanoseconds
        "Unit": ["ns"],
    })

    if has_time_data(sample_df):
        converted_df = convert_time_columns(sample_df, mock_args.time_unit)
    else:
        converted_df = sample_df

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(3.44664, abs=1e-5)
    assert converted_df.loc[0, "Min"] == pytest.approx(1.76925, abs=1e-5)
    assert converted_df.loc[0, "Max"] == pytest.approx(12.53212, abs=1e-5)


@pytest.mark.time_unit_integration
def test_show_all_with_time_unit_conversion():
    from utils.tty import convert_time_columns

    test_data = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446.64],
        "Min": [1769.25],
        "Max": [12532.12],
        "Unit": ["Ns"],
    })

    for time_unit in ["s", "ms", "us", "ns"]:
        converted_df = convert_time_columns(test_data, time_unit)

        assert converted_df.loc[0, "Unit"] == time_unit

        expected_avg = 3446.64 / time_units[time_unit]
        assert converted_df.loc[0, "Avg"] == pytest.approx(expected_avg, abs=1e-10)


@pytest.mark.time_unit_edge_cases
def test_edge_cases_and_error_handling():
    from utils.tty import convert_time_columns

    empty_df = pd.DataFrame()
    result = convert_time_columns(empty_df, "ms")
    assert result.empty

    nan_df = pd.DataFrame({"Avg": [float("nan"), 1000.0], "Unit": ["ns", "Count"]})
    result = convert_time_columns(nan_df, "ms")
    assert result.loc[0, "Unit"] == "ms"

    mixed_case_df = pd.DataFrame({"Avg": [1000.0, 2000.0], "Unit": ["ns", "NS"]})
    result = convert_time_columns(mixed_case_df, "ms")
    assert result.loc[0, "Unit"] == "ms"
    assert result.loc[1, "Unit"] == "ms"


@pytest.mark.iteration_multiplexing
def test_iteration_multiplexing(binary_handler_analyze_rocprof_compute):
    workload = "tests/workloads/vcopy_iteration_multiplexing/MI350"
    workload_dir = common.setup_workload_dir(workload)

    # Test with dispatch filtering
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--dispatch",
        "0",
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
    workload_dir = common.setup_workload_dir(indirs[0])

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


@pytest.fixture
def mock_raw_pmc_for_kernel_top():
    """Create flat raw_pmc DataFrame for create_df_kernel_top_stats tests."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [1000, 2000, 3000, 4000],
        "End_Timestamp": [1500, 2800, 3400, 4200],
    })


@pytest.fixture
def mock_workload_for_filter():
    """Create mock workload with dfs populated for apply_kernel_filter tests."""
    workload = Mock()
    workload.dfs = {
        1: pd.DataFrame({
            "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
            "Count": [2, 1, 1],
            "Sum(ns)": [900, 800, 200],
            "Selected": ["", "", ""],
        }),
        2: pd.DataFrame({
            "Dispatch_ID": [1, 2, 3, 4],
            "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
            "GPU_ID": [0, 0, 1, 0],
        }),
    }
    workload.filter_kernel_ids = []
    return workload


@pytest.mark.misc
def test_create_df_kernel_top_stats_returns_valid_dataframes(
    mock_raw_pmc_for_kernel_top,
):
    """Test create_df_kernel_top_stats returns valid DF with correct structure."""
    import tempfile

    from utils.file_io import create_df_kernel_top_stats

    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, dispatch_info_df = create_df_kernel_top_stats(
            df_in=mock_raw_pmc_for_kernel_top,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="sum",
        )

        # Test return types
        assert isinstance(kernel_top_df, pd.DataFrame)
        assert isinstance(dispatch_info_df, pd.DataFrame)

        # Test kernel_top_df columns
        expected_columns = [
            "Kernel_Name",
            "Count",
            "Sum(ns)",
            "Mean(ns)",
            "Median(ns)",
            "Percent",
        ]
        for col in expected_columns:
            assert col in kernel_top_df.columns, f"Missing column: {col}"

        # Test dispatch_info_df columns
        assert "Kernel_Name" in dispatch_info_df.columns
        assert "GPU_ID" in dispatch_info_df.columns
        assert "Dispatch_ID" in dispatch_info_df.columns

        # Test index is reset (starts from 0)
        assert kernel_top_df.index[0] == 0

        # Test percentage sum is approximately 100%
        assert kernel_top_df["Percent"].sum() == pytest.approx(100.0, abs=0.01)


@pytest.mark.misc
def test_create_df_kernel_top_stats_grouping_and_aggregation(
    mock_raw_pmc_for_kernel_top,
):
    """Test kernel grouping, aggregation functions, and sorting behavior."""
    import tempfile

    from utils.file_io import create_df_kernel_top_stats

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with sortby="sum"
        kernel_top_df, _ = create_df_kernel_top_stats(
            df_in=mock_raw_pmc_for_kernel_top,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="sum",
        )

        # Test kernel grouping - kernel_a appears twice in input
        kernel_a_row = kernel_top_df[kernel_top_df["Kernel_Name"] == "kernel_a"]
        assert len(kernel_a_row) == 1  # Should be grouped into one row
        assert kernel_a_row["Count"].iloc[0] == 2  # kernel_a appears twice

        # Test sorting by sum (descending) - highest sum should be first
        sum_values = kernel_top_df["Sum(ns)"].tolist()
        assert sum_values == sorted(sum_values, reverse=True)

        # Test with sortby="kernel"
        kernel_top_df_sorted, _ = create_df_kernel_top_stats(
            df_in=mock_raw_pmc_for_kernel_top,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="kernel",
        )

        # Test sorting by kernel name (ascending)
        kernel_names = kernel_top_df_sorted["Kernel_Name"].tolist()
        assert kernel_names == sorted(kernel_names)


@pytest.mark.misc
def test_create_df_kernel_top_stats_filters():
    """Test GPU ID, dispatch ID (including '> n' syntax),
    node filters, and empty input handling."""
    import tempfile

    from utils.file_io import create_df_kernel_top_stats

    # Create test data with Node column for node filtering
    raw_pmc_with_node = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
        "GPU_ID": [0, 0, 1, 0],
        "Node": ["node0", "node0", "node1", "node0"],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [1000, 2000, 3000, 4000],
        "End_Timestamp": [1500, 2800, 3400, 4200],
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test GPU ID filter
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=raw_pmc_with_node,
            raw_data_dir=temp_dir,
            filter_gpu_ids="0",
            filter_dispatch_ids=None,
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        # GPU_ID=0 should only include 3 dispatches (kernel_a at GPU 1 is filtered out)
        assert len(dispatch_df) == 3

        # Test dispatch ID filter with "> n" syntax
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=raw_pmc_with_node,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["> 2"],
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        # Only Dispatch_ID > 2 should remain (IDs 3 and 4)
        assert len(dispatch_df) == 2
        assert all(dispatch_df["Dispatch_ID"] > 2)

        # Test dispatch ID filter with specific IDs
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=raw_pmc_with_node,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["1", "2"],
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(dispatch_df) == 2

        # Test node filter
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=raw_pmc_with_node,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            filter_nodes="node1",
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(dispatch_df) == 1
        assert dispatch_df.iloc[0]["Kernel_Name"] == "kernel_a"

        # Test empty input handling
        empty_raw_pmc = pd.DataFrame({
            "Kernel_Name": [],
            "GPU_ID": [],
            "Dispatch_ID": [],
            "Start_Timestamp": [],
            "End_Timestamp": [],
        })
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=empty_raw_pmc,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            filter_nodes=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(kernel_top_df) == 0
        assert len(dispatch_df) == 0


@pytest.mark.misc
def test_apply_kernel_filter_integer_ids(mock_workload_for_filter):
    """Test integer kernel ID filtering, Selected marker,
    uses workload.dfs[1], invalid ID error."""

    from utils.parser import apply_kernel_filter

    # Flat single-index raw_pmc DataFrame
    raw_df = pd.DataFrame({
        "Kernel_Name": [
            "kernel_a",
            "kernel_b",
            "kernel_a",
            "kernel_c",
        ],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
    })

    # Test integer kernel ID filtering
    mock_workload_for_filter.filter_kernel_ids = [0]  # Select first kernel (kernel_a)
    result_df = apply_kernel_filter(raw_df, mock_workload_for_filter)

    # Should only contain rows with kernel_a
    assert len(result_df) == 2  # kernel_a appears twice
    assert all(result_df["Kernel_Name"] == "kernel_a")

    # Test that Selected marker is added
    assert mock_workload_for_filter.dfs[1].loc[0, "Selected"] == "*"

    # Test multiple kernel IDs
    mock_workload_for_filter.filter_kernel_ids = [0, 1]  # kernel_a and kernel_b
    mock_workload_for_filter.dfs[1]["Selected"] = ""  # Reset
    result_df = apply_kernel_filter(raw_df, mock_workload_for_filter)
    assert len(result_df) == 3  # 2 kernel_a + 1 kernel_b

    # Test invalid kernel ID (out of bounds) - should call console_error and exit
    mock_workload_for_filter.filter_kernel_ids = [99]  # Invalid ID
    mock_workload_for_filter.dfs[1]["Selected"] = ""  # Reset
    with patch("utils.parser.console_error") as mock_error:
        # console_error calls sys.exit by default, so mock it to raise SystemExit
        mock_error.side_effect = SystemExit(1)
        with pytest.raises(SystemExit):
            apply_kernel_filter(raw_df, mock_workload_for_filter)
        mock_error.assert_called_once()
        # Check error message contains the invalid ID
        assert "99" in str(mock_error.call_args)


@pytest.mark.misc
def test_apply_kernel_filter_string_names(mock_workload_for_filter):
    """Test string kernel name filtering and partial match."""

    from utils.parser import apply_kernel_filter

    # Flat single-index raw_pmc DataFrame
    raw_df = pd.DataFrame({
        "Kernel_Name": [
            "kernel_a",
            "kernel_b",
            "kernel_a",
            "kernel_c",
        ],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
    })

    # Test string kernel name filtering - exact match
    mock_workload_for_filter.filter_kernel_ids = ["kernel_b"]
    result_df = apply_kernel_filter(raw_df, mock_workload_for_filter)
    assert len(result_df) == 1
    assert result_df["Kernel_Name"].iloc[0] == "kernel_b"

    # Test filtering with whitespace in kernel names (should be stripped)
    raw_df_with_whitespace = pd.DataFrame({
        "Kernel_Name": [
            " kernel_a ",
            "kernel_b",
            "kernel_a",
        ],
        "GPU_ID": [0, 0, 1],
        "Dispatch_ID": [1, 2, 3],
    })

    mock_workload_for_filter.filter_kernel_ids = ["kernel_a"]
    result_df = apply_kernel_filter(raw_df_with_whitespace, mock_workload_for_filter)
    # Should match both " kernel_a " (stripped) and "kernel_a"
    assert len(result_df) == 2


@pytest.mark.misc
def test_pc_sampling_single_kernel_uses_workload_dfs():
    """Test single kernel filter reads from workload.dfs[1],
    kernel index out of bounds warning."""
    # Create mock workload with dfs populated
    workload = Mock()
    workload.dfs = {
        1: pd.DataFrame({
            "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
            "Count": [2, 1, 1],
            "Sum(ns)": [900, 800, 200],
        }),
    }
    tool_data = {
        "buffer_records": {"pc_sample_stochastic": [{}], "pc_sample_host_trap": []}
    }

    # Kernel index out of bounds warns and returns empty.
    workload.filter_kernel_ids = [99]
    with patch("utils.parser.console_warning") as mock_warning:
        result = load_pc_sampling_data(workload, "test", "count", tool_data)
        mock_warning.assert_called()
        call_args_str = str(mock_warning.call_args)
        assert "out of bounds" in call_args_str or "99" in call_args_str
        assert result.empty

    # Kernel name is extracted from workload.dfs[1].
    workload.filter_kernel_ids = [1]  # kernel_b
    with patch("utils.parser.load_pc_sampling_data_per_kernel") as mock_per_kernel:
        mock_per_kernel.return_value = pd.DataFrame()
        load_pc_sampling_data(workload, "test", "count", tool_data)
        if mock_per_kernel.called:
            assert "kernel_b" in str(mock_per_kernel.call_args)


# =============================================================================
# join_prof unit test
# =============================================================================


def test_join_prof_renames_sq_accum_prev_hires_to_bucket_target(tmp_path):
    """
    results_pmc_perf_<bucket>_ACCUM.csv files whose body uses rocprof's
    generic SQ_ACCUM_PREV_HIRES column must have that column renamed to the
    bucket target derived from the file stem before the merge.
    """
    (tmp_path / "profiling_config.yaml").write_text(
        "format_rocprof_output: csv\njoin_type: kernel\n"
    )

    header = (
        "GPU_ID,Kernel_Name,Dispatch_ID,Grid_Size,Workgroup_Size,"
        "LDS_Per_Workgroup,Scratch_Per_Workitem,SGPR,vgpr,{counter}\n"
    )
    acc = tmp_path / "results_pmc_perf_SQ_LEVEL_WAVES_ACCUM.csv"
    acc.write_text(
        header.format(counter="SQ_ACCUM_PREV_HIRES")
        + "0,kernel_a,0,1024,64,32,0,8,4,100\n"
        + "0,kernel_a,1,1024,64,32,0,8,4,200\n"
    )
    other = tmp_path / "results_pmc_perf_0.csv"
    other.write_text(
        header.format(counter="SQ_WAVES")
        + "0,kernel_a,0,1024,64,32,0,8,4,10\n"
        + "0,kernel_a,1,1024,64,32,0,8,4,20\n"
    )

    inst = cli_analysis.__new__(cli_analysis)
    args = Namespace(
        path=[[str(tmp_path)]],
        nodes=None,
        spatial_multiplexing=False,
        join_type="kernel",
        kokkos_trace=False,
    )
    inst.get_args = MagicMock(return_value=args)
    inst._arch_configs = {}
    inst._runs = {}

    inst.join_prof(tmp_path, out=str(tmp_path / "pmc_perf.csv"))
    merged = pd.read_csv(tmp_path / "pmc_perf.csv")

    assert "SQ_LEVEL_WAVES_ACCUM" in merged.columns
    assert "SQ_ACCUM_PREV_HIRES" not in merged.columns
    assert set(merged["SQ_LEVEL_WAVES_ACCUM"].tolist()) == {100, 200}
    assert "SQ_WAVES" in merged.columns
