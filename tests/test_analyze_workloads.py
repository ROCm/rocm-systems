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

from unittest.mock import patch

import pytest

##################################################
##          Generated tests                     ##
##################################################


def test_analyze_vcopy_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/vcopy/MI100"]
    )
    assert code == 0


def test_analyze_vcopy_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/vcopy/MI200"]
    )
    assert code == 0


def test_analyze_vcopy_MI300_rocpd(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/vcopy/MI300X_A1_rocpd"]
    )
    assert code == 0


def test_analyze_ipblocks_TCP_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCP/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TCP_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCP/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_TCP_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCP/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TCP_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCP/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQC/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQC_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQC/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQC/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQC_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQC/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_HBM_LDS_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_HBM_LDS/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_TCC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCC/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TCC_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCC/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_TCC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCC/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TCC_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TCC/MI200"]
    )
    assert code == 0


def test_analyze_no_roof_MI350(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/no_roof/MI350"]
    )
    assert code == 0


def test_analyze_no_roof_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/no_roof/MI300X_A1"]
    )
    assert code == 0


def test_analyze_no_roof_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/no_roof/MI100"]
    )
    assert code == 0


def test_analyze_no_roof_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/no_roof/MI300A_A1"]
    )
    assert code == 0


def test_analyze_no_roof_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/no_roof/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_CPC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_CPC/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_CPC_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_CPC/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_CPC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_CPC/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_CPC_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_CPC/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_0_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_0_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0/MI100"]
    )
    assert code == 0


def test_analyze_dispatch_0_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_0_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0/MI200"]
    )
    assert code == 0


def test_analyze_join_type_grid_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_grid/MI300X_A1"]
    )
    assert code == 0


def test_analyze_join_type_grid_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_grid/MI100"]
    )
    assert code == 0


def test_analyze_join_type_grid_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_grid/MI300A_A1"]
    )
    assert code == 0


def test_analyze_join_type_grid_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_grid/MI200"]
    )
    assert code == 0


def test_analyze_kernel_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel/MI300X_A1"]
    )
    assert code == 0


def test_analyze_kernel_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel/MI100"]
    )
    assert code == 0


def test_analyze_kernel_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel/MI300A_A1"]
    )
    assert code == 0


def test_analyze_kernel_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel/MI200"]
    )
    assert code == 0


def test_analyze_kernel_substr_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_substr/MI300X_A1"]
    )
    assert code == 0


def test_analyze_kernel_substr_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_substr/MI100"]
    )
    assert code == 0


def test_analyze_kernel_substr_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_substr/MI300A_A1"]
    )
    assert code == 0


def test_analyze_kernel_substr_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_substr/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_7_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_7/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_7_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_7/MI100"]
    )
    assert code == 1


def test_analyze_dispatch_7_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_7/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_7_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_7/MI200"]
    )
    assert code == 1


def test_analyze_kernel_inv_int_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_int/MI300X_A1"]
    )
    assert code == 0


def test_analyze_kernel_inv_int_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_int/MI100"]
    )
    assert code == 1


def test_analyze_kernel_inv_int_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_int/MI300A_A1"]
    )
    assert code == 0


def test_analyze_kernel_inv_int_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_int/MI200"]
    )
    assert code == 1


def test_analyze_mem_levels_vL1D_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_vL1D/MI200"]
    )
    assert code == 0


def test_analyze_sort_kernels_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/sort_kernels/MI200"]
    )
    assert code == 0


def test_analyze_kernel_inv_str_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_str/MI300X_A1"]
    )
    assert code == 0


def test_analyze_kernel_inv_str_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_str/MI100"]
    )
    assert code == 1


def test_analyze_kernel_inv_str_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_str/MI300A_A1"]
    )
    assert code == 0


def test_analyze_kernel_inv_str_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_inv_str/MI200"]
    )
    assert code == 1


def test_analyze_ipblocks_SQ_SPI_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_2_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_2/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_2_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_2/MI100"]
    )
    assert code == 0


def test_analyze_dispatch_2_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_2/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_2_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_2/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_0_1_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0_1/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_0_1_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0_1/MI100"]
    )
    assert code == 0


def test_analyze_dispatch_0_1_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0_1/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_0_1_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_0_1/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_LDS_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_LDS/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_TA_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TA/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TA_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TA/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_TA_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TA/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TA_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TA/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_6_8_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_6_8/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_6_8_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_6_8/MI100"]
    )
    assert code == 1


def test_analyze_dispatch_6_8_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_6_8/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_6_8_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_6_8/MI200"]
    )
    assert code == 1


def test_analyze_device_inv_int_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_inv_int/MI300X_A1"]
    )
    assert code == 0


def test_analyze_device_inv_int_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_inv_int/MI100"]
    )
    assert code == 0


def test_analyze_device_inv_int_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_inv_int/MI300A_A1"]
    )
    assert code == 0


def test_analyze_device_inv_int_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_inv_int/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_TA_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_TA/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_TA_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_TA/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_TA_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_TA/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_TA_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_TA/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_TD_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TD/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TD_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TD/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_TD_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TD/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_TD_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_TD/MI200"]
    )
    assert code == 0


def test_analyze_device_filter_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_filter/MI300X_A1"]
    )
    assert code == 0


def test_analyze_device_filter_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_filter/MI100"]
    )
    assert code == 0


def test_analyze_device_filter_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_filter/MI300A_A1"]
    )
    assert code == 0


def test_analyze_device_filter_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/device_filter/MI200"]
    )
    assert code == 0


def test_analyze_join_type_kernel_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_kernel/MI300X_A1"]
    )
    assert code == 0


def test_analyze_join_type_kernel_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_kernel/MI100"]
    )
    assert code == 0


def test_analyze_join_type_kernel_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_kernel/MI300A_A1"]
    )
    assert code == 0


def test_analyze_join_type_kernel_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/join_type_kernel/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300X_A1(
    binary_handler_analyze_rocprof_compute,
):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300A_A1(
    binary_handler_analyze_rocprof_compute,
):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_L2_vL1d_LDS_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_L2_vL1d_LDS/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_CPF_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPF/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_CPF_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPF/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_CPF_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPF/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_CPF_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPF/MI200"]
    )
    assert code == 0


def test_analyze_sort_dispatches_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/sort_dispatches/MI200"]
    )
    assert code == 0


def test_analyze_kernel_names_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/kernel_names/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_vL1d_LDS_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_vL1d_LDS/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_L2_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_L2/MI200"]
    )
    assert code == 0


def test_analyze_dispatch_inv_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_inv/MI300X_A1"]
    )
    assert code == 0


def test_analyze_dispatch_inv_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_inv/MI100"]
    )
    assert code == 0


def test_analyze_dispatch_inv_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_inv/MI300A_A1"]
    )
    assert code == 0


def test_analyze_dispatch_inv_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/dispatch_inv/MI200"]
    )
    assert code == 0


def test_analyze_path_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/path/MI300X_A1"]
    )
    assert code == 0


def test_analyze_path_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/path/MI100"]
    )
    assert code == 0


def test_analyze_path_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/path/MI300A_A1"]
    )
    assert code == 0


def test_analyze_path_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/path/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_CPC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPC/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_CPC_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPC/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_CPC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPC/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_CPC_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_CPC/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300X_A1(
    binary_handler_analyze_rocprof_compute,
):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300A_A1(
    binary_handler_analyze_rocprof_compute,
):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI200"]
    )
    assert code == 0


def test_analyze_mem_levels_HBM_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/mem_levels_HBM/MI200"]
    )
    assert code == 0


def test_analyze_ipblocks_SPI_MI300X_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SPI/MI300X_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SPI_MI100(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SPI/MI100"]
    )
    assert code == 0


def test_analyze_ipblocks_SPI_MI300A_A1(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SPI/MI300A_A1"]
    )
    assert code == 0


def test_analyze_ipblocks_SPI_MI200(binary_handler_analyze_rocprof_compute):
    code = binary_handler_analyze_rocprof_compute(
        ["analyze", "--path", "tests/workloads/ipblocks_SPI/MI200"]
    )
    assert code == 0
