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


from importlib.machinery import SourceFileLoader
from unittest.mock import patch

import pytest

rocprof_compute = SourceFileLoader("rocprof-compute", "src/rocprof-compute").load_module()

##################################################
##          Generated tests                     ##
##################################################


def test_import_D_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_dev01p3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dev01p3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_SQC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SQC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_SQC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SQC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_Axes2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_Axes2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_SQ_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SQ/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_SQ_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SQ/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_CPF_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/CPF/mi100",
        ],
    ):
        rocprof_compute.main()


def test_CPF_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/CPF/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_LDS_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_LDS/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_LDS_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_LDS/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_str_inv4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_str_inv4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_SPI_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SPI/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_SPI_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SPI/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_valid_2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_valid_2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_str_valid_2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_valid_2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_mixbench1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_mixbench1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_mixbench1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_mixbench1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_TA_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TA/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_TA_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TA/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_CPF_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_CPF/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_CPF_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_CPF/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_LDS_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/LDS/mi100",
        ],
    ):
        rocprof_compute.main()


def test_LDS_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/LDS/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_valid_3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_valid_3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_SQC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_SQC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_Axes2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_Axes2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_HBM_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/HBM/mi100",
        ],
    ):
        rocprof_compute.main()


def test_HBM_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/HBM/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_TA_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_TA_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_val_int_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_val_int/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_L2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_L2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_L2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_L2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_L2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/L2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_L2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/L2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_dev1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dev1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_dev1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dev1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_str_valid_1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_str_valid_1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_Axes3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_Axes3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_TD_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TD/mi100",
        ],
    ):
        rocprof_compute.main()


def test_TD_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TD/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_val_int2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_val_int2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_val_int2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_val_int2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_mixbench2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_mixbench2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_mixbench2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_mixbench2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_SPI_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_SPI/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_val_int2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_val_int2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_val_int2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_val_int2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_TA_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_TA/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_valid_3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_valid_3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_SQ_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SQ/mi100",
        ],
    ):
        rocprof_compute.main()


def test_SQ_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SQ/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_dev01p3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dev01p3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_val_int2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_val_int2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_str_inv4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_str_inv4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_CPF_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_CPF/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_mixbench_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench/mi100",
        ],
    ):
        rocprof_compute.main()


def test_mixbench_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_str_inv4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_str_inv4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_kernels_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_kernels/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_kernels_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_kernels/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_TCC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_TCC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_TA_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TA_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_TA_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TA_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_SQ_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_SQ/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_TCP_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_TCP/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_str_valid_2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_str_valid_2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_Axes3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_Axes3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_dev0_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dev0/mi100",
        ],
    ):
        rocprof_compute.main()


def test_dev0_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dev0/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_Axes1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_Axes1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_HBM_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_HBM/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_val_int_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_val_int/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_val_int_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_val_int/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_TCC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TCC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_TCC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TCC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_SQC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SQC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_SQC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_SQC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_TD_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_TD/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_Axes1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_Axes1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_SPI_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SPI/mi100",
        ],
    ):
        rocprof_compute.main()


def test_SPI_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/SPI/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_val_int_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_val_int/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_val_int_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_val_int/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_dispatches_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_dispatches/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_mixbench2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_mixbench2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_Axes4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_Axes4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_TCP_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TCP/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_TCP_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TCP/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_LDS_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_LDS/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_invdev_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/invdev/mi100",
        ],
    ):
        rocprof_compute.main()


def test_invdev_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/invdev/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_dev0_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dev0/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_dev0_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dev0/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_Axes1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_Axes1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_invdev_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_invdev/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_kernels_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/kernels/mi100",
        ],
    ):
        rocprof_compute.main()


def test_kernels_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/kernels/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_Axes4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_Axes4_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Axes4/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_CMD_INV_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/CMD_INV/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_mixbench2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_mixbench2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_Double_N_flag_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_Double_N_flag/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_TD_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TD/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_TD_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TD/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_TCC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TCC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_TCC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TCC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_dev0_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_dev0/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_str_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_str_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_L2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_L2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_TA_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TA_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_TA_CPC_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_TA_CPC/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_Double_N_flag_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Double_N_flag/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_Double_N_flag_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_Double_N_flag/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_Double_N_flag_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Double_N_flag/mi100",
        ],
    ):
        rocprof_compute.main()


def test_Double_N_flag_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Double_N_flag/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_K_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_K_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_valid_1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_valid_1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_str_valid_1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_valid_1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_mixbench1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_mixbench1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_dev1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dev1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_dev1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dev1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_valid_1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_valid_1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_str_valid_1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_valid_1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_mixbench1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_mixbench1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/mixbench1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_CMD_INV_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_CMD_INV/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_D_str_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_D_str_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_HBM_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_HBM/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_HBM_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_HBM/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_kernels_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_kernels/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_K_str_valid_2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_valid_2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_K_str_valid_2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/K_str_valid_2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_TCP_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TCP/mi100",
        ],
    ):
        rocprof_compute.main()


def test_TCP_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TCP/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_Axes3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_Axes3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/Axes3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_invdev_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_invdev/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_invdev_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_invdev/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_dispatches_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dispatches/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_dispatches_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_dispatches/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_D_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_D_str_inv3_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/D_str_inv3/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_TA_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TA/mi100",
        ],
    ):
        rocprof_compute.main()


def test_TA_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/TA/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_D_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_D_int_inv1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_D_int_inv1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_dispatches_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dispatches/mi100",
        ],
    ):
        rocprof_compute.main()


def test_dispatches_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/dispatches/mi100",
        ],
    ):
        rocprof_compute.main()


def test_roof_only_dev1_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/roof_only_dev1/mi100",
        ],
    ):
        rocprof_compute.main()


def test_import_no_roof_K_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "temp",
            "-p",
            "temp123",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()


def test_no_roof_K_int_inv2_mi100():
    with patch(
        "sys.argv",
        [
            "rocprof-compute",
            "database",
            "--import",
            "-H",
            "localhost",
            "-u",
            "-p",
            "temp123",
            "temp",
            "-t",
            "asw",
            "-w",
            "tests/workloads/no_roof_K_int_inv2/mi100",
        ],
    ):
        rocprof_compute.main()
