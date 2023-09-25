import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import pandas as pd

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()
workload_1 = os.path.realpath("workload")
app = ["./sample/vcopy", "1048576", "256"]
ALL_CSVS = [
    "pmc_dispatch_info.csv",
    "pmc_kernel_top.csv",
    "roofline.csv",
    "SQ_IFETCH_LEVEL.csv",
    "SQ_INST_LEVEL_LDS.csv",
    "SQ_INST_LEVEL_SMEM.csv",
    "SQ_INST_LEVEL_VMEM.csv",
    "SQ_LEVEL_WAVES.csv",
    "sysinfo.csv",
    "timestamps.csv",
]


def test_path():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--",
            ]
            + app,
        ):
            omniperf.main()
    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernel",
                "kernel_name",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_summaries():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-summaries",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_summaries():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-summaries",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQC",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_TA():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TA",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_TD():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TD",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_TCP():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCP",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_TCC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCC",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SPI():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SPI",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPC",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_CPF():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPF",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "CPC",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ_TA():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "TA",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ_SPI():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SPI",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ_SQC_TCP_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SQC",
                "TCP",
                "CPC",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_ipblocks_SQ_SPI_TA_TCC_CPF():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SPI",
                "TA",
                "TCC",
                "CPF",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_dispatch_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_dispatch_0_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0",
                "1",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_dispatch_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "2",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "0",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "1",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "2",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "3",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "4",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_kernel_verbose_5():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "5",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_join_type_grid():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "grid",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_join_type_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "kernel",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_device_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--device",
                "0",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_no_roof():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--no-roof",
                "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_sort_dispatches():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--sort",
                "dispatches," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_sort_kernels():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--sort",
                "kernels," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_HBM():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "HBM," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_L2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "L2," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_vL1D():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "vL1D," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "LDS," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_HBM_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "HBM",
                "LDS," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_vL1D_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "vL1D",
                "LDS," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS


def test_mem_levels_L2_vL1D_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "-VVV",
                "--path",
                workload_1,
                "--mem-levels",
                "L2",
                "vL1D",
                "LDS," "--",
            ]
            + app,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    files_in_workload = os.listdir(workload_1)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(file)
            assert len(file_dict[file].index)

    assert file_dict.keys() == ALL_CSVS
