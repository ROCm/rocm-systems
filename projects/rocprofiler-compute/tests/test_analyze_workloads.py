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
import pytest
from unittest.mock import patch
from importlib.machinery import SourceFileLoader

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()

##################################################
##          Generated tests                     ##
##################################################ule()


def test_analyze_dispatch_invalid_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_invalid/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_4_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_4/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_no_roof_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/no_roof/MI100"]
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_5_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_5/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_CPC/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI100",
            ],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_2_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_2/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQ_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SQ/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_substr_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_substr/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_CPF_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_CPF/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_dispatch_0_1_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_0_1/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_0_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_0/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SQC/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_join_type_grid_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/join_type_grid/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_TCP_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_TCP/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_3_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_3/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_device_inv_int_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/device_inv_int/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQ_SPI_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SQ_SPI/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_summaries_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_summaries/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_dispatch_2_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_2/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_inv_int_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_inv_int/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 1


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "analyze",
                "--path",
                "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100",
            ],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_dispatch_7_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_7/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 1


def test_analyze_kernel_inv_str_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_inv_str/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 1


def test_analyze_ipblocks_TCC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_TCC/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQ_TA_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SQ_TA/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SPI_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SPI/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_SQ_CPC_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_SQ_CPC/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_dispatch_6_8_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_6_8/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 1


def test_analyze_dispatch_0_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/dispatch_0/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_kernel_verbose_1_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/kernel_verbose_1/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_TA_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_TA/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_ipblocks_TD_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/ipblocks_TD/MI100"],
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_path_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv", ["omniperf", "analyze", "--path", "tests/workloads/path/MI100"]
        ):
            omniperf.main()

    assert e.value.code == 0


def test_analyze_join_type_kernel_MI100():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            ["omniperf", "analyze", "--path", "tests/workloads/join_type_kernel/MI100"],
        ):
            omniperf.main()
