import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader
import pandas as pd
import subprocess
import re
import shutil
import inspect
import sys
import test_utils
from unittest.mock import patch
from importlib.machinery import SourceFileLoader

rocprof_compute = SourceFileLoader("rocprof-compute", "src/rocprof-compute").load_module()

##################################################
##          Generated tests                     ##
##################################################


def test_analyze_vcopy_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/vcopy/MI100"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_vcopy_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/vcopy/MI200"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCP_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCP/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCP_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCP/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCP_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCP/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCP_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCP/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQC_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQC/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQC/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQC_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQC/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQC_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQC/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_HBM_LDS_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_HBM_LDS/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCC_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCC/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCC/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCC_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCC/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TCC_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TCC/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_no_roof_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/no_roof/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_no_roof_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/no_roof/MI100"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_no_roof_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/no_roof/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_no_roof_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/no_roof/MI200"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_CPC_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_CPC/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_CPC/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_CPC_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_CPC/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_CPC_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_CPC/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_grid_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_grid/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_grid_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_grid/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_grid_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_grid/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_grid_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_grid/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/kernel/MI100"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/kernel/MI200"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_substr_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_substr/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_substr_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_substr/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_substr_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_substr/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_substr_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_substr/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_7_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_7/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_7_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_7/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_dispatch_7_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_7/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_7_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_7/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_kernel_inv_int_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_int/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_inv_int_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_int/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_kernel_inv_int_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_int/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_inv_int_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_int/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_mem_levels_vL1D_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_vL1D/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_sort_kernels_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/sort_kernels/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_inv_str_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_str/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_inv_str_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_str/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_kernel_inv_str_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_str/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_inv_str_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_inv_str/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_ipblocks_SQ_SPI_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_2_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_2/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_2_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_2/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_2_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_2/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_2_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_2/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_1_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0_1/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_1_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0_1/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_1_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0_1/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_0_1_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_0_1/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_LDS_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_LDS/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TA_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TA/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TA_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TA/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TA_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TA/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TA_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TA/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_6_8_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_6_8/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_6_8_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_6_8/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_dispatch_6_8_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_6_8/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_6_8_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_6_8/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 1


def test_analyze_device_inv_int_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_inv_int/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_inv_int_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_inv_int/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_inv_int_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_inv_int/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_inv_int_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_inv_int/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_TA_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_TA/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_TA_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_TA/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_TA_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_TA/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_TA_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_TA/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TD_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TD/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TD_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TD/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TD_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TD/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_TD_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_TD/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_filter_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_filter/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_filter_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_filter/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_filter_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_filter/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_device_filter_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/device_filter/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_kernel_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_kernel/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_kernel_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_kernel/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_kernel_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_kernel/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_join_type_kernel_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/join_type_kernel/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_L2_vL1d_LDS_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_L2_vL1d_LDS/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPF_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPF/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPF_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPF/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPF_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPF/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPF_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPF/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_sort_dispatches_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/sort_dispatches/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_kernel_names_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/kernel_names/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_vL1d_LDS_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_vL1d_LDS/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_L2_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_L2/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_inv_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_inv/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_inv_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_inv/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_inv_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_inv/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_dispatch_inv_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/dispatch_inv/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_path_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/path/MI300X_A1"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_path_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/path/MI100"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_path_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/path/MI300A_A1"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_path_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["rocprof-compute", "analyze", "--path", "tests/workloads/path/MI200"],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPC_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPC/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPC/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPC_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPC/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_CPC_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_CPC/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_mem_levels_HBM_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/mem_levels_HBM/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SPI_MI300X_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SPI/MI300X_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SPI_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SPI/MI100",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SPI_MI300A_A1():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SPI/MI300A_A1",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0


def test_analyze_ipblocks_SPI_MI200():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "rocprof-compute",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SPI/MI200",
            ],
        ):
            rocprof_compute.main()
    assert e.value.code == 0
