import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import shutil
import pandas as pd

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()

baseline_opts = ["omniperf", "analyze"]


@pytest.mark.misc
def test_valid_path():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/vcopy/MI100"],
        ):
            omniperf.main()
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/vcopy/MI200"],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.misc
def test_list_kernels():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--list-stats",
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
                "--list-stats",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.list_metrics
def test_list_metrics_gfx90a():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx90a"]):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--list-metrics",
                "gfx90a",
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
                "--list-metrics",
                "gfx90a",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.list_metrics
def test_list_metrics_gfx906():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx906"]):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--list-metrics",
                "gfx906",
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
                "--list-metrics",
                "gfx906",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.list_metrics
def test_list_metrics_gfx908():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx908"]):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--list-metrics",
                "gfx908",
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
                "--list-metrics",
                "gfx908",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "1",
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
                "--block",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "5",
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
                "--block",
                "5",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "5.2.2",
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
                "--block",
                "5.2.2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "6.1",
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
                "--block",
                "6.1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_5():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "10",
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
                "--block",
                "10",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_block
def test_filter_block_6():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--block",
                "100",
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
                "--block",
                "100",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_kernel
def test_filter_kernel_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel",
                "0",
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
                "--kernel",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.filter_kernel
def test_filter_kernel_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--kernel",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1


@pytest.mark.filter_kernel
def test_filter_kernel_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel",
                "0",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--kernel",
                "0",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1


@pytest.mark.dispatch
def test_dispatch_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dispatch",
                "0",
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
                "--dispatch",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.dispatch
def test_dispatch_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dispatch",
                "1",
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
                "--dispatch",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.dispatch
def test_dispatch_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dispatch",
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
                "tests/workloads/vcopy/MI200",
                "--dispatch",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.dispatch
def test_dispatch_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dispatch",
                "1",
                "4",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--dispatch",
                "1",
                "4",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1


@pytest.mark.dispatch
def test_dispatch_5():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dispatch",
                "5",
                "6",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI200",
                "--dispatch",
                "5",
                "6",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1


@pytest.mark.misc
def test_gpu_ids():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--gpu-id",
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
                "tests/workloads/vcopy/MI200",
                "--gpu-id",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.normal_unit
def test_normal_unit_per_wave():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--normal-unit",
                "per_wave",
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
                "--normal-unit",
                "per_wave",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.normal_unit
def test_normal_unit_per_cycle():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--normal-unit",
                "per_cycle",
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
                "--normal-unit",
                "per_cycle",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.normal_unit
def test_normal_unit_per_second():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--normal-unit",
                "per_second",
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
                "--normal-unit",
                "per_second",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.normal_unit
def test_normal_unit_per_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--normal-unit",
                "per_kernel",
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
                "--normal-unit",
                "per_kernel",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.max_stat
def test_max_stat_num_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--max-stat-num",
                "0",
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
                "--max-stat-num",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.max_stat
def test_max_stat_num_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--max-stat-num",
                "5",
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
                "--max-stat-num",
                "5",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.max_stat
def test_max_stat_num_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--max-stat-num",
                "10",
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
                "--max-stat-num",
                "10",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.max_stat
def test_max_stat_num_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--max-stat-num",
                "15",
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
                "--max-stat-num",
                "15",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.time_unit
def test_time_unit_s():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--time-unit",
                "s",
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
                "--time-unit",
                "s",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.time_unit
def test_time_unit_ms():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--time-unit",
                "ms",
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
                "--time-unit",
                "ms",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.time_unit
def test_time_unit_us():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--time-unit",
                "us",
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
                "--time-unit",
                "us",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.time_unit
def test_time_unit_ns():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--time-unit",
                "ns",
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
                "--time-unit",
                "ns",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.decimal
def test_decimal_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--decimal",
                "0",
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
                "--decimal",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.decimal
def test_decimal_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--decimal",
                "1",
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
                "--decimal",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.decimal
def test_decimal_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--decimal",
                "4",
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
                "--decimal",
                "4",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.misc
def test_save_dfs():
    output_path = "tests/workloads/vcopy/saved_analysis"

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--save-dfs",
                output_path,
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

    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
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


@pytest.mark.col
def test_col_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--cols",
                "0",
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
                "--cols",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.col
def test_col_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--cols",
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
                "tests/workloads/vcopy/MI200",
                "--cols",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.col
def test_col_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--cols",
                "0",
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
                "tests/workloads/vcopy/MI200",
                "--cols",
                "0",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.misc
def test_g():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "-g",
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
                "-g",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "0",
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
                "--kernel-verbose",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "1",
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
                "--kernel-verbose",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
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
                "tests/workloads/vcopy/MI200",
                "--kernel-verbose",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "3",
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
                "--kernel-verbose",
                "3",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "4",
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
                "--kernel-verbose",
                "4",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_5():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "5",
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
                "--kernel-verbose",
                "5",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


@pytest.mark.kernel_verbose
def test_kernel_verbose_6():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--kernel-verbose",
                "6",
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
                "--kernel-verbose",
                "6",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


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
    assert e.value.code == 1

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
    assert e.value.code == 1


@pytest.mark.misc
def test_dependency_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/vcopy/MI100",
                "--dependency",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0
