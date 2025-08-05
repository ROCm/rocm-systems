##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################


import os
import shutil
from unittest.mock import Mock

import pandas as pd
import pytest
import test_utils

config = {}
config["cleanup"] = True if "PYTEST_XDIST_WORKER_COUNT" in os.environ else False

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
    "tests/workloads/vcopy/MI300X_A1_rocpd",
    "tests/workloads/vcopy/MI350",
]

time_units = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


@pytest.mark.misc
def test_valid_path(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_list_kernels(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--list-stats",
            ]
        )
        assert code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx90a(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--list-metrics",
            "gfx90a",
        ]
    )
    assert code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--list-metrics",
                "gfx90a",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--list-metrics",
            "gfx908",
        ]
    )
    assert code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--list-metrics",
                "gfx908",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "1",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "5",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "5.2.2",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "6.1",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "10",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_6(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "100",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "1",
            ]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel",
                "0",
                "1",
            ]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "1",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "2",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "1",
                "4",
            ]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dispatch",
                "5",
                "6",
            ]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        # if dir.endswith("MI350") or dir.endswith("MI300X_A1_rocpd"):
        if dir in (
            "tests/workloads/vcopy/MI350",
            "tests/workloads/vcopy/MI300X_A1_rocpd",
        ):
            gpu_id = "0"
        else:
            gpu_id = "2"
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--gpu-id",
                gpu_id,
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_wave(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_wave",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_cycle(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_cycle",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_second(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_second",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_kernel(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--normal-unit",
                "per_kernel",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "5",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "10",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--max-stat-num",
                "15",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_s(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "s",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ms(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "ms",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_us(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "us",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ns(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--time-unit",
                "ns",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "1",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--decimal",
                "4",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_save_dfs(binary_handler_analyze_rocprof_compute):
    output_path = "tests/workloads/vcopy/saved_analysis"
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--save-dfs",
                output_path,
            ]
        )
        assert code == 0

        files_in_workload = os.listdir(output_path)
        for file_name in files_in_workload:
            df = pd.read_csv(output_path + "/" + file_name)
            assert len(df.index) >= 1

        shutil.rmtree(output_path)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "2",
                "--include-cols",
                "Description",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--cols",
                "0",
                "2",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_g(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "-g",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_0(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "0",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "1",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "2",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "3",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "4",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "5",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_6(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--kernel-verbose",
                "6",
            ]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_baseline(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI200",
            "--path",
            "tests/workloads/vcopy/MI100",
        ]
    )
    assert code == 0

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI200",
            "--path",
            "tests/workloads/vcopy/MI200",
        ]
    )
    assert code == 1

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            "tests/workloads/vcopy/MI100",
            "--path",
            "tests/workloads/vcopy/MI100",
        ]
    )
    assert code == 1


# =============================================================================
# Test cases for Parser.py
# =============================================================================


@pytest.mark.misc
def test_dependency_MI100(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            [
                "analyze",
                "--path",
                workload_dir,
                "--dependency",
            ]
        )
        assert code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_parser_utility_functions():
    """Test parser utility functions edge cases"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import numpy as np
    import pandas as pd

    from utils.parser import (
        to_concat,
        to_int,
        to_max,
        to_median,
        to_min,
        to_mod,
        to_quantile,
        to_round,
        to_std,
    )

    try:
        result = to_min(None, None)
        assert np.isnan(result), "to_min with all None should return nan"
    except TypeError:
        pass

    try:
        result = to_min(None, 5)
        assert False, "Should have crashed"
    except TypeError:
        pass

    result = to_min(7, 3, 9, 1)
    assert result == 1, "to_min should return minimum value"

    try:
        result = to_max(None, None)
        assert np.isnan(result), "to_max with all None should return nan"
    except TypeError:
        pass

    try:
        result = to_max(None, 5)
        assert False, "Should have crashed"
    except TypeError:
        pass

    result = to_max(7, 3, 9, 1)
    assert result == 9, "to_max should return maximum value"

    result = to_median(None)
    assert result is None, "to_median should return None for None input"

    try:
        to_median("invalid_string")
        assert False, "to_median should raise exception for invalid type"
    except Exception as e:
        assert "unsupported type" in str(e)

    try:
        to_std("invalid_string")
        assert False, "to_std should raise exception for invalid type"
    except Exception as e:
        assert "unsupported type" in str(e)

    result = to_int(None)
    assert result is None, "to_int should return None for None input"

    try:
        to_int(["list", "not", "supported"])
        assert False, "to_int should raise exception for invalid type"
    except Exception as e:
        assert "unsupported type" in str(e)

    result = to_quantile(None, 0.5)
    assert result is None, "to_quantile should return None for None input"

    try:
        to_quantile("invalid_string", 0.5)
        assert False, "to_quantile should raise exception for invalid type"
    except Exception as e:
        assert "unsupported type" in str(e)

    result = to_concat("hello", "world")
    assert result == "helloworld", "to_concat should concatenate strings"

    result = to_concat(123, 456)
    assert result == "123456", "to_concat should convert to strings and concatenate"

    series = pd.Series([1.234, 2.567, 3.890])
    result = to_round(series, 2)
    expected = pd.Series([1.23, 2.57, 3.89])
    pd.testing.assert_series_equal(result, expected)

    result = to_round(3.14159, 2)
    assert result == 3.14, "to_round should round scalar values"

    series = pd.Series([10, 15, 20])
    result = to_mod(series, 3)
    expected = pd.Series([1, 0, 2])
    pd.testing.assert_series_equal(result, expected)

    result = to_mod(10, 3)
    assert result == 1, "to_mod should return modulo for scalars"


@pytest.mark.misc
def test_parser_error_handling():
    """Test parser error handling paths"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from utils.parser import build_eval_string, calc_builtin_var, update_denom_string

    try:
        build_eval_string("AVG(SQ_WAVES)", None, config={})
        assert False, "Should have raised exception for None coll_level"
    except Exception as e:
        assert "coll_level can not be None" in str(e)

    assert build_eval_string("", "pmc_perf", config={}) == ""
    assert update_denom_string("", "per_wave") == ""

    class MockSysInfo:
        total_l2_chan = 32

    sys_info = MockSysInfo()
    try:
        calc_builtin_var("$unsupported_var", sys_info)
        assert False, "Should have raised exception for unsupported var"
    except SystemExit:
        pass


@pytest.mark.misc
def test_parser_error_handling():
    """Test parser error handling paths"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from utils.parser import build_eval_string, calc_builtin_var, update_denom_string

    try:
        build_eval_string("AVG(SQ_WAVES)", None, config={})
        assert False, "Should have raised exception for None coll_level"
    except Exception as e:
        assert "coll_level can not be None" in str(e)

    assert build_eval_string("", "pmc_perf", config={}) == ""
    assert update_denom_string("", "per_wave") == ""

    class MockSysInfo:
        total_l2_chan = 32

    sys_info = MockSysInfo()
    try:
        calc_builtin_var("$unsupported_var", sys_info)
        assert False, "Should have raised exception for unsupported var"
    except SystemExit:
        pass


@pytest.mark.misc
def test_missing_file_handling(binary_handler_analyze_rocprof_compute):
    """Test handling of missing files"""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        code = binary_handler_analyze_rocprof_compute(["analyze", "--path", temp_dir])
        assert code != 0


@pytest.mark.misc
def test_ast_transformer_edge_cases():
    """Simplified test focusing on the actual code paths"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import ast

    from utils.parser import CodeTransformer

    transformer = CodeTransformer()

    unknown_call = ast.Call(
        func=ast.Name(id="UNKNOWN_FUNCTION", ctx=ast.Load()),
        args=[ast.Constant(value=5) if hasattr(ast, "Constant") else ast.Num(n=5)],
        keywords=[],
    )

    try:
        result = transformer.visit_Call(unknown_call)
        if hasattr(result.func, "id") and result.func.id == "UNKNOWN_FUNCTION":
            assert False, "Function name should have been changed or exception raised"
    except Exception as e:
        assert "Unknown call" in str(
            e
        ), f"Expected 'Unknown call' in error, got: {str(e)}"

    supported_call = ast.Call(
        func=ast.Name(id="MIN", ctx=ast.Load()),
        args=[ast.Constant(value=5) if hasattr(ast, "Constant") else ast.Num(n=5)],
        keywords=[],
    )

    try:
        result = transformer.visit_Call(supported_call)
        assert result.func.id == "to_min", f"Expected 'to_min', got: {result.func.id}"
    except Exception as e:
        assert False, f"Supported function call should not raise exception: {e}"


@pytest.mark.misc
def test_analyze_with_debug_mode(binary_handler_analyze_rocprof_compute):
    """Test analyze to cover debug paths in eval_metric - using direct function call"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import pandas as pd

    from utils.parser import eval_metric

    mock_dfs = {
        1: pd.DataFrame(
            {
                "Metric_ID": ["1.1.0"],
                "Metric": ["Test Metric"],
                "Expr": ["AVG(SQ_WAVES)"],
                "coll_level": ["pmc_perf"],
            }
        ).set_index("Metric_ID")
    }

    mock_dfs_type = {1: "metric_table"}

    class MockSysInfo:
        ip_blocks = "standard"
        se_per_gpu = 4
        pipes_per_gpu = 4
        cu_per_gpu = 64
        simd_per_cu = 4
        sqc_per_gpu = 16
        lds_banks_per_cu = 32
        cur_sclk = 1800.0
        cur_mclk = 1200.0
        max_sclk = 2100.0
        max_mclk = 1600.0
        max_waves_per_cu = 40
        num_hbm_channels = 4
        total_l2_chan = 32
        num_xcd = 1
        wave_size = 64

    sys_info = MockSysInfo()

    raw_pmc_df = {
        "pmc_perf": pd.DataFrame(
            {
                "SQ_WAVES": [100, 200, 150],
                "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
                "End_Timestamp": [1000000, 2000000, 1500000],
                "Start_Timestamp": [0, 1000000, 500000],
            }
        )
    }

    try:
        eval_metric(
            mock_dfs, mock_dfs_type, sys_info, raw_pmc_df, debug=True, config={}
        )
    except Exception:
        pass


@pytest.mark.misc
def test_filter_combinations_coverage(binary_handler_analyze_rocprof_compute):
    """Test basic filters that should work"""
    for dir in ["tests/workloads/vcopy/MI100", "tests/workloads/vcopy/MI200"]:
        if os.path.exists(dir):
            workload_dir = test_utils.setup_workload_dir(dir)

            code = binary_handler_analyze_rocprof_compute(
                [
                    "analyze",
                    "--path",
                    workload_dir,
                ]
            )
            assert code == 0

            code = binary_handler_analyze_rocprof_compute(
                [
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "SQ",
                ]
            )
            assert code == 0

            test_utils.clean_output_dir(config["cleanup"], workload_dir)
            break


@pytest.mark.misc
def test_apply_filters_direct():
    """Test apply_filters function directly to cover filter branches"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import pandas as pd

    from utils.parser import apply_filters

    class MockWorkload:
        def __init__(self):
            self.raw_pmc = pd.DataFrame(
                {
                    ("pmc_perf", "GPU_ID"): [0, 0, 1, 1],
                    ("pmc_perf", "Kernel_Name"): [
                        "vecCopy",
                        "vecAdd",
                        "vecCopy",
                        "vecMul",
                    ],
                    ("pmc_perf", "Dispatch_ID"): [0, 1, 2, 3],
                    ("pmc_perf", "Node"): ["node0", "node0", "node1", "node1"],
                }
            )
            self.raw_pmc.columns = pd.MultiIndex.from_tuples(self.raw_pmc.columns)

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


@pytest.mark.misc
def test_missing_files_scenarios(binary_handler_analyze_rocprof_compute):
    """Test scenarios with missing files to cover error paths"""
    import shutil
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

                code = binary_handler_analyze_rocprof_compute(
                    [
                        "analyze",
                        "--path",
                        workload_dir,
                    ]
                )
            break


@pytest.mark.misc
def test_pc_sampling_basic_coverage():
    """Test PC sampling functions with minimal data"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import tempfile

    from utils.parser import load_pc_sampling_data, search_pc_sampling_record

    class MockWorkload:
        filter_kernel_ids = []

    workload = MockWorkload()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = load_pc_sampling_data(workload, temp_dir, "none", "count")
        assert result.empty

        result = load_pc_sampling_data(workload, temp_dir, "missing", "count")
        assert result.empty

        workload.filter_kernel_ids = [0, 1, 2]  # Multiple kernels
        result = load_pc_sampling_data(workload, temp_dir, "test", "count")
        assert result.empty

        result = search_pc_sampling_record([])
        assert result is None


@pytest.mark.misc
def test_build_dfs_edge_cases():
    """Test build_dfs and gen_counter_list with various configurations"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from utils.parser import gen_counter_list

    visited, counters = gen_counter_list(None)
    assert not visited
    assert counters == []

    visited, counters = gen_counter_list(123)
    assert not visited
    assert counters == []

    visited, counters = gen_counter_list("AVG(SQ_WAVES + TCC_HIT)")
    assert visited
    assert "SQ_WAVES" in counters
    assert "TCC_HIT" in counters

    visited, counters = gen_counter_list("Start_Timestamp + End_Timestamp")
    assert visited

    visited, counters = gen_counter_list("INVALID SYNTAX !!!")
    assert not visited


@pytest.mark.misc
def test_update_functions_coverage():
    """Test update_denom_string and update_normUnit_string branches"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from utils.parser import update_denom_string, update_normUnit_string

    result = update_denom_string("AVG(SQ_WAVES / $denom)", "per_wave")
    assert "$denom" not in result
    assert "SQ_WAVES" in result

    result = update_denom_string("AVG(DATA / $denom)", "per_cycle")
    assert "$GRBM_GUI_ACTIVE_PER_XCD" in result

    result = update_denom_string("AVG(DATA / $denom)", "per_second")
    assert "End_Timestamp - Start_Timestamp" in result

    result = update_denom_string("AVG(DATA / $denom)", "unsupported_unit")
    assert "$denom" in result

    result = update_normUnit_string("(Prefix + $normUnit)", "per_wave")
    assert "per wave" in result.lower()
    assert result[0].isupper()


@pytest.fixture
def sample_time_data():
    return pd.DataFrame(
        {
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
        }
    )


@pytest.fixture
def original_ns_values():
    return {"Avg": 3446.64, "Min": 1769.25, "Max": 12532.12}


@pytest.mark.time_unit_conversion
def test_has_time_data_detection(sample_time_data):
    from utils.tty import has_time_data

    assert has_time_data(sample_time_data)

    no_time_data = pd.DataFrame(
        {
            "Metric": ["Non-Time Metric"],
            "Avg": [1000.0],
            "Unit": ["Count"],
        }
    )
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

    assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-10
    assert abs(converted_df.loc[0, "Min"] - expected_min) < 1e-10
    assert abs(converted_df.loc[0, "Max"] - expected_max) < 1e-10

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

    assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-6
    assert abs(converted_df.loc[0, "Min"] - expected_min) < 1e-6
    assert abs(converted_df.loc[0, "Max"] - expected_max) < 1e-6


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_microseconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "us")

    assert converted_df.loc[0, "Unit"] == "us"

    expected_avg = original_ns_values["Avg"] / time_units["us"]
    expected_min = original_ns_values["Min"] / time_units["us"]
    expected_max = original_ns_values["Max"] / time_units["us"]

    assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-3
    assert abs(converted_df.loc[0, "Min"] - expected_min) < 1e-3
    assert abs(converted_df.loc[0, "Max"] - expected_max) < 1e-3


@pytest.mark.time_unit_conversion
def test_time_unit_conversion_to_nanoseconds(sample_time_data, original_ns_values):
    from utils.tty import convert_time_columns

    converted_df = convert_time_columns(sample_time_data, "ns")

    assert converted_df.loc[0, "Unit"] == "ns"

    assert abs(converted_df.loc[0, "Avg"] - original_ns_values["Avg"]) < 1e-10
    assert abs(converted_df.loc[0, "Min"] - original_ns_values["Min"]) < 1e-10
    assert abs(converted_df.loc[0, "Max"] - original_ns_values["Max"]) < 1e-10


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
    assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-6


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

        assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-10
        assert abs(converted_df.loc[0, "Min"] - expected_min) < 1e-10
        assert abs(converted_df.loc[0, "Max"] - expected_max) < 1e-10
        assert converted_df.loc[0, "Unit"] == target_unit


# Integration tests with show_all functionality
@pytest.mark.time_unit_integration
def test_integration_conversion_flow():
    from utils.tty import convert_time_columns, has_time_data

    mock_args = Mock()
    mock_args.time_unit = "ms"
    mock_args.decimal = 2

    sample_df = pd.DataFrame(
        {
            "Metric_ID": ["7.2.0"],
            "Metric": ["Kernel Time"],
            "Avg": [3446640.0],  # 3.44664 ms in nanoseconds
            "Min": [1769250.0],  # 1.76925 ms in nanoseconds
            "Max": [12532120.0],  # 12.53212 ms in nanoseconds
            "Unit": ["ns"],
        }
    )

    if has_time_data(sample_df):
        converted_df = convert_time_columns(sample_df, mock_args.time_unit)
    else:
        converted_df = sample_df

    assert converted_df.loc[0, "Unit"] == "ms"
    assert abs(converted_df.loc[0, "Avg"] - 3.44664) < 1e-5
    assert abs(converted_df.loc[0, "Min"] - 1.76925) < 1e-5
    assert abs(converted_df.loc[0, "Max"] - 12.53212) < 1e-5


@pytest.mark.time_unit_integration
def test_show_all_with_time_unit_conversion():
    from utils.tty import convert_time_columns

    test_data = pd.DataFrame(
        {
            "Metric_ID": ["7.2.0"],
            "Metric": ["Kernel Time"],
            "Avg": [3446.64],
            "Min": [1769.25],
            "Max": [12532.12],
            "Unit": ["Ns"],
        }
    )

    for time_unit in ["s", "ms", "us", "ns"]:
        converted_df = convert_time_columns(test_data, time_unit)

        assert converted_df.loc[0, "Unit"] == time_unit

        expected_avg = 3446.64 / time_units[time_unit]
        assert abs(converted_df.loc[0, "Avg"] - expected_avg) < 1e-10


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
