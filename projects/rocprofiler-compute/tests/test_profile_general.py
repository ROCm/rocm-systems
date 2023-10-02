import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import pandas as pd
import subprocess
import re

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()
workload_1 = os.path.realpath("workload")
kernel_name_1 = "vecCopy(double*, double*, double*, int, int) [clone .kd]"
app_1 = ["./sample/vcopy", "1048576", "256"]
ALL_CSVS = [
    "SQ_IFETCH_LEVEL.csv",
    "SQ_INST_LEVEL_LDS.csv",
    "SQ_INST_LEVEL_SMEM.csv",
    "SQ_INST_LEVEL_VMEM.csv",
    "SQ_LEVEL_WAVES.csv",
    "pmc_perf.csv",
    "pmc_perf_0.csv",
    "pmc_perf_1.csv",
    "pmc_perf_10.csv",
    "pmc_perf_11.csv",
    "pmc_perf_12.csv",
    "pmc_perf_13.csv",
    "pmc_perf_14.csv",
    "pmc_perf_15.csv",
    "pmc_perf_16.csv",
    "pmc_perf_2.csv",
    "pmc_perf_3.csv",
    "pmc_perf_4.csv",
    "pmc_perf_5.csv",
    "pmc_perf_6.csv",
    "pmc_perf_7.csv",
    "pmc_perf_8.csv",
    "pmc_perf_9.csv",
    "sysinfo.csv",
    "timestamps.csv",
]


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "rocm-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load rocm-smi")
        assert 0
    return p.stdout.decode("ascii")


def gpu_soc():
    rocminfo = str(
        subprocess.run(
            ["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.decode("ascii")
    )
    rocminfo = rocminfo.split("\n")
    soc_regex = re.compile(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", re.MULTILINE)
    gpu_id = list(filter(soc_regex.match, rocminfo))[0].split()[1]

    print("gpu_id", gpu_id)
    if gpu_id == "gfx906":
        return "mi50"
    elif gpu_id == "gfx908":
        return "mi100"
    elif gpu_id == "gfx90a":
        return "mi200"
    elif gpu_id == "gfx900":
        return "vega10"
    else:
        print("Invalid SoC")
        assert 0


soc = gpu_soc()


def test_path():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            print("length is: ", len(file_dict[file].index))
            print(file_dict[file])
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel",
                kernel_name_1,
                "kernel_name",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_summaries():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernel-summaries",
                "vcopy",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_ipblocks_SQ():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_TA():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TA",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_TD():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TD",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_TCP():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCP",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_TCC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "TCC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_10.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SPI():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SPI",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_CPF():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "CPF",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQ_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "CPC",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQ_TA():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "TA",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQ_SPI():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--ipblocks",
                "SQ",
                "SPI",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQ_SQC_TCP_CPC():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
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
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_ipblocks_SQ_SPI_TA_TCC_CPF():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
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
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)

    print(sorted(list(file_dict.keys())))
    assert sorted(list(file_dict.keys())) == [
        "SQ_IFETCH_LEVEL.csv",
        "SQ_INST_LEVEL_LDS.csv",
        "SQ_INST_LEVEL_SMEM.csv",
        "SQ_INST_LEVEL_VMEM.csv",
        "SQ_LEVEL_WAVES.csv",
        "pmc_perf.csv",
        "pmc_perf_0.csv",
        "pmc_perf_1.csv",
        "pmc_perf_10.csv",
        "pmc_perf_2.csv",
        "pmc_perf_3.csv",
        "pmc_perf_4.csv",
        "pmc_perf_5.csv",
        "pmc_perf_6.csv",
        "pmc_perf_7.csv",
        "pmc_perf_8.csv",
        "pmc_perf_9.csv",
        "sysinfo.csv",
        "timestamps.csv",
    ]


def test_dispatch_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_dispatch_0_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "0",
                "1",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_dispatch_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--dispatch",
                "2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "1",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_3():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "3",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_4():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "4",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_verbose_5():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--kernelVerbose",
                "5",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_join_type_grid():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "grid",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_join_type_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--join-type",
                "kernel",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_device_0():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--device",
                "0",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_no_roof():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--no-roof",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_sort_dispatches():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--sort",
                "dispatches",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_sort_kernels():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--sort",
                "kernels",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_HBM():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "HBM",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_L2():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "L2",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_vL1D():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "vL1D",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_HBM_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "HBM",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_vL1D_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "vL1D",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return

    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_mem_levels_L2_vL1D_LDS():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--mem-level",
                "L2",
                "vL1D",
                "LDS",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()
    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return
    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS


def test_kernel_names():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app_1",
                "-VVV",
                "--path",
                workload_1,
                "--roof-only",
                "--kernel-names",
                "--",
            ]
            + app_1,
        ):
            omniperf.main()

    if soc == "mi100":
        # assert that it did not run
        assert e.value.code == 2
        # Do not continue testing
        return
    # assert successful run
    assert e.value.code == 0
    workload_1_dir = workload_1 + "/app_1/" + soc
    files_in_workload = os.listdir(workload_1_dir)

    # Check if csvs have data
    file_dict = {}
    for file in files_in_workload:
        if file.endswith(".csv"):
            file_dict[file] = pd.read_csv(workload_1_dir + "/" + file)
            assert len(file_dict[file].index)
    assert sorted(list(file_dict.keys())) == ALL_CSVS
