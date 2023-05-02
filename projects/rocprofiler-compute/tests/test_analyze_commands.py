import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()


def test_valid_path_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench/mi100"],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_inv_path_mi100():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--path", "workloads/vpaste"]):
            omniperf.main()
    assert e.value.code == 2


def test_No_flags_mi100():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze"]):
            omniperf.main()
    assert e.value.code == 2


def test_List_Kernels_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--list-kernels",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_List_metrics_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx90a"]):
            omniperf.main()
    assert e.value.code == 0


def test_List_metrics_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx906"]):
            omniperf.main()
    assert e.value.code == 0


def test_List_metrics_3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--list-metrics", "gfx908"]):
            omniperf.main()
    assert e.value.code == 0


def test_filter_metrics_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-b",
                "SQ, LDS",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_metrics_inv_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-b",
                "Crash, Test",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-k",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_inv_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-k",
                "99",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_filter_dispatch_ids_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--dispatch",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_dispatch_ids_inv_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--dispatch",
                "99",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_ids_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--gpu-id",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_ids_inv_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--gpu-id",
                "99",
            ],
        ):
            omniperf.main()
        assert e.value.code == 1


def test_select_t_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-t",
                "s",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-t",
                "ms",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-t",
                "us",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_4_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "-t",
                "ns",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_dec_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--decimal",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_dec_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--decimal",
                "16",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_col_1_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--cols",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_col_2_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--cols",
                "SQ",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_col_3_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--cols",
                "inv",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_dependency_mi100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi100",
                "--dependency",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_valid_path_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/mixbench/mi200"],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_inv_path_mi200():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze", "--path", "workloads/vpaste"]):
            omniperf.main()
    assert e.value.code == 2


def test_No_flags_mi200():
    with pytest.raises(SystemExit) as e:
        with patch("sys.argv", ["omniperf", "analyze"]):
            omniperf.main()
    assert e.value.code == 2


def test_List_Kernels_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--list-kernels",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_metrics_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-b",
                "SQ, LDS",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_metrics_inv_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-b",
                "Crash, Test",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-k",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_inv_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-k",
                "99",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_filter_dispatch_ids_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--dispatch",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_dispatch_ids_inv_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--dispatch",
                "99",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_ids_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--gpu-id",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_filter_gpu_ids_inv_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--gpu-id",
                "99",
            ],
        ):
            omniperf.main()
    assert e.value.code == 1


def test_select_t_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-t",
                "s",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-t",
                "ms",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-t",
                "us",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_select_t_4_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "-t",
                "ns",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_dec_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--decimal",
                "0",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_dec_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--decimal",
                "16",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_col_1_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--cols",
                "2",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0


def test_col_2_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--cols",
                "SQ",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_col_3_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--cols",
                "inv",
            ],
        ):
            omniperf.main()
    assert e.value.code == 2


def test_dependency_mi200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/mixbench/mi200",
                "--dependency",
            ],
        ):
            omniperf.main()
    assert e.value.code == 0
