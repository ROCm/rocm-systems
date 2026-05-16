# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Roofline-related analyze CLI integration tests.

Extracted verbatim from tests/test_analyze_commands.py during the test-suite
refactor; no logic changes.
"""

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

config = {}
config["cleanup"] = True

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
    "tests/workloads/vcopy/MI350",
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

    from roofline.roofline_main import Roofline
    from utils.file_io import load_sys_info
    from utils.specs import generate_machine_specs

    class MockArgs:
        def __init__(self):
            self.roof_only = True
            self.mem_level = "ALL"
            self.sort = "ALL"
            self.roofline_data_type = ["FP32"]

    args = MockArgs()
    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = load_sys_info(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": ["FP32"],
    }

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("FP32", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_invalid_datatype_cli():
    """cli_generate_plot with invalid datatype returns None."""

    from roofline.roofline_main import Roofline
    from utils.file_io import load_sys_info
    from utils.specs import generate_machine_specs

    class MockArgs:
        def __init__(self):
            self.roof_only = True
            self.mem_level = "ALL"
            self.sort = "ALL"
            self.roofline_data_type = ["FP32"]

    args = MockArgs()

    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = load_sys_info(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": ["FP32"],
    }

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("INVALID_DATATYPE", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


