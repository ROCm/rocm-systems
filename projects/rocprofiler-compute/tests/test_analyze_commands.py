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
config["cleanup"] = True

indir1 = "tests/workloads/vcopy/MI100"
indir2 = "tests/workloads/vcopy/MI200"


@pytest.mark.misc
def test_valid_path():
    workload_dir = test_utils.setup_workload_dir(indir1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", workload_dir],
        ):
            omniperf.main()
    assert e.value.code == 0

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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

    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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

    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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

    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                indir1,
                "--decimal",
                "1",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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

    workload_dir = test_utils.setup_workload_dir(indir1)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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
    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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

    workload_dir = test_utils.setup_workload_dir(indir2)
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
    workload_dir = test_utils.setup_workload_dir(indir1)
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
