import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import shutil
import pandas as pd
import test_utils

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()

baseline_opts = ["omniperf", "analyze"]

config = {}
config["cleanup"] = True if "PYTEST_XDIST_WORKER_COUNT" in os.environ else False

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
]


@pytest.mark.misc
def test_valid_path():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                ["omniperf", "analyze", "--path", workload_dir],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_list_kernels():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--list-stats",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx90a():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx90a"]):
            omniperf.main()
    assert e.value.code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--list-metrics",
                    "gfx90a",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx906():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx906"]):
            omniperf.main()
    assert e.value.code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--list-metrics",
                    "gfx906",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.list_metrics
def test_list_metrics_gfx908():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx908"]):
            omniperf.main()
    assert e.value.code == 1

    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--list-metrics",
                    "gfx908",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "5",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "5.2.2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "6.1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "10",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.filter_block
def test_filter_block_6():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "100",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_filter_kernel_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel",
                    "0",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dispatch",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dispatch",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dispatch",
                    "2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dispatch",
                    "1",
                    "4",
                ],
            ):
                omniperf.main()
        assert e.value.code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.serial
def test_dispatch_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dispatch",
                    "5",
                    "6",
                ],
            ):
                omniperf.main()
        assert e.value.code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_gpu_ids():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--gpu-id",
                    "2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_wave():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--normal-unit",
                    "per_wave",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_cycle():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--normal-unit",
                    "per_cycle",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_second():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--normal-unit",
                    "per_second",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.normal_unit
def test_normal_unit_per_kernel():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--normal-unit",
                    "per_kernel",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--max-stat-num",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--max-stat-num",
                    "5",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--max-stat-num",
                    "10",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.max_stat
def test_max_stat_num_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--max-stat-num",
                    "15",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_s():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--time-unit",
                    "s",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ms():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--time-unit",
                    "ms",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_us():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--time-unit",
                    "us",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.time_unit
def test_time_unit_ns():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--time-unit",
                    "ns",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--decimal",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--decimal",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.decimal
def test_decimal_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--decimal",
                    "4",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_save_dfs():
    output_path = "tests/workloads/vcopy/saved_analysis"
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--save-dfs",
                    output_path,
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

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
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                workload_dir,
                "--save-dfs",
                output_path,
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    files_in_workload = os.listdir(output_path)
    for file_name in files_in_workload:
        df = pd.read_csv(output_path + "/" + file_name)
        if file_name in single_row_tables:
            assert len(df.index) == 1
        else:
            assert len(df.index) >= 3
    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--cols",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--cols",
                    "2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.col
def test_col_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--cols",
                    "0",
                    "2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_g():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "-g",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_0():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "0",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_1():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "1",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_2():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "2",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_3():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "3",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_4():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "4",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_5():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "5",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.kernel_verbose
def test_kernel_verbose_6():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--kernel-verbose",
                    "6",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.misc
def test_baseline():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--path",
                "tests/workloads/vcopy/MI100",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--path",
                "tests/workloads/vcopy/MI200",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--path",
                "tests/workloads/vcopy/MI100",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/multikernel/MI200",
                "-k",
                "0",
                "--path",
                "tests/workloads/multikernel/MI200",
                "-k",
                "1",
                "--path",
                "tests/workloads/multikernel/MI200",
                "-k",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/multikernel/MI200",
                "-k",
                "0",
                "--path",
                "tests/workloads/multikernel/MI200",
                "-k",
                "1",
                "--path",
                "tests/workloads/vcopy/MI100",
                "-k",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.misc
def test_dependency_MI100():
    for dir in indirs:
        workload_dir = test_utils.setup_workload_dir(dir)
        with pytest.raises(SystemExit) as e:
            with patch(
                "sys.argv",
                [
                    "omniperf",
                    "analyze",
                    "--path",
                    workload_dir,
                    "--dependency",
                ],
            ):
                omniperf.main()
        assert e.value.code == 0
    test_utils.clean_output_dir(config["cleanup"], workload_dir)
