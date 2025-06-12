##############################################################################bl
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import os
import shutil
from importlib.machinery import SourceFileLoader
from unittest.mock import patch

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
    "tests/workloads/vcopy/MI350",
]


@pytest.mark.misc
def test_valid_path(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_list_kernels(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--list-stats"]
        )
        assert code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx90a(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(["analyze", "--list-metrics", "gfx90a"])
    assert code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--list-metrics", "gfx90a"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx906(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(["analyze", "--list-metrics", "gfx906"])
    assert code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--list-metrics", "gfx906"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(["analyze", "--list-metrics", "gfx908"])
    assert code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--list-metrics", "gfx908"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "1"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "5"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "5.2.2"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "6.1"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "10"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_6(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--block", "100"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel", "1"]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel", "0", "1"]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dispatch", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dispatch", "1"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dispatch", "2"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dispatch", "1", "4"]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dispatch", "5", "6"]
        )
        assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        if dir.endswith("MI350"):
            gpu_id = "0"
        else:
            gpu_id = "2"
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--gpu-id", gpu_id]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_wave(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--normal-unit", "per_wave"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_cycle(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--normal-unit", "per_cycle"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_second(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--normal-unit", "per_second"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_kernel(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--normal-unit", "per_kernel"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--max-stat-num", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--max-stat-num", "5"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--max-stat-num", "10"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--max-stat-num", "15"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_s(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--time-unit", "s"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ms(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--time-unit", "ms"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_us(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--time-unit", "us"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ns(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--time-unit", "ns"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--decimal", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--decimal", "1"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--decimal", "4"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_save_dfs(binary_handler_analyze_rocprof_compute):
    output_path = "tests/workloads/vcopy/saved_analysis"
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--save-dfs", output_path]
        )
        assert code == 0

        files_in_workload = os.listdir(output_path)
        single_row_tables = [
            "0.1_Top_Kernels.csv",
            "13.3_Instruction_Cache_-_L2_Interface.csv",
            "18.1_Aggregate_Stats_(All_channels).csv",
        ]
        for file_name in files_in_workload:
            df = pd.read_csv(output_path + "/" + file_name)
            if file_name in single_row_tables:
                assert len(df.index) == 1
            else:
                assert len(df.index) >= 3

        shutil.rmtree(output_path)
    test_utils.clean_output_dir(config["cleanup"], workload_dir)

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", workload_dir, "--save-dfs", output_path]
    )
    assert code == 0

    files_in_workload = os.listdir(output_path)
    for file_name in files_in_workload:
        df = pd.read_csv(output_path + "/" + file_name)
        if file_name in single_row_tables:
            assert len(df.index) == 1
        else:
            assert len(df.index) >= 3
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--cols", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--cols", "2"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--cols", "0", "2"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_g(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "-g"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_0(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "0"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_1(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "1"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_2(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "2"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_3(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "3"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_4(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "4"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_5(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "5"]
        )
        assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_6(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--kernel-verbose", "6"]
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


@pytest.mark.misc
def test_dependency_MI100(binary_handler_analyze_rocprof_compute):
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        code = binary_handler_analyze_rocprof_compute(
            ["analyze", "--path", workload_dir, "--dependency"]
        )
        assert code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)
