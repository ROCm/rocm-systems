# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

##################################################
##          Generated tests                     ##
##################################################

import os

import test_utils

config = {}
config["cleanup"] = True if "PYTEST_XDIST_WORKER_COUNT" in os.environ else False


def test_analyze_vcopy_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/vcopy/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_vcopy_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/vcopy/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCP_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TCP/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCP_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TCP/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCP_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TCP/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCP_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TCP/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQC/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQC_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQC/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQC/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQC_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQC/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_HBM_LDS_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/mem_levels_HBM_LDS/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TCC/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCC_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TCC/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TCC/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TCC_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TCC/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_no_roof_MI350(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/no_roof/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_no_roof_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/no_roof/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_no_roof_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/no_roof/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_no_roof_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/no_roof/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_no_roof_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/no_roof/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_CPC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_CPC/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_CPC_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_CPC/MI100"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_CPC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_CPC/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_CPC_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_CPC/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_grid_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_grid/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_grid_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/join_type_grid/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_grid_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_grid/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_grid_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/join_type_grid/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_substr_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_substr/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_substr_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_substr/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_substr_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_substr/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_substr_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_substr/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_7_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_7/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_7_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_7/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_7_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_7/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_7_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_7/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_int_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_inv_int/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_int_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_inv_int/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_int_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_inv_int/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_int_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_inv_int/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_vL1D_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/mem_levels_vL1D/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_sort_kernels_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/sort_kernels/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_str_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_inv_str/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_str_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_inv_str/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_str_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/kernel_inv_str/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_inv_str_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_inv_str/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI/MI100"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_2_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_2/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_2_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_2/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_2_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_2/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_2_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_2/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_1_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_0_1/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_1_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0_1/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_1_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_0_1/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_0_1_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_0_1/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_LDS_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/mem_levels_LDS/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TA_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TA/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TA_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TA/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TA_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TA/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TA_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TA/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_6_8_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_6_8/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_6_8_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_6_8/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_6_8_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_6_8/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_6_8_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_6_8/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 1

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_inv_int_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/device_inv_int/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_inv_int_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/device_inv_int/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_inv_int_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/device_inv_int/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_inv_int_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/device_inv_int/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_TA_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_TA/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_TA_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQ_TA/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_TA_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_TA/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_TA_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQ_TA/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TD_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TD/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TD_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TD/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TD_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_TD/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_TD_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_TD/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_filter_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/device_filter/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_filter_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/device_filter/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_filter_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/device_filter/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_device_filter_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/device_filter/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_kernel_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_kernel/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_kernel_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_kernel/MI100"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_kernel_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_kernel/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_join_type_kernel_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/join_type_kernel/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300X_A1(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI100"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI300A_A1(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SQC_TCP_CPC_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SQC_TCP_CPC/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_L2_vL1d_LDS_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/mem_levels_L2_vL1d_LDS/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPF_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_CPF/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPF_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_CPF/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPF_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_CPF/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPF_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_CPF/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_sort_dispatches_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/sort_dispatches/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_kernel_names_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/kernel_names/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_vL1d_LDS_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/mem_levels_vL1d_LDS/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQ/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SQ/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_L2_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/mem_levels_L2/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_inv_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_inv/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_inv_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_inv/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_inv_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/dispatch_inv/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_dispatch_inv_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/dispatch_inv/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_path_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/path/MI300X_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_path_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/path/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_path_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/path/MI300A_A1")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_path_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/path/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPC_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_CPC/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPC_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_CPC/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPC_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_CPC/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_CPC_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_CPC/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300X_A1(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI100(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI100"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI300A_A1(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SQ_SPI_TA_TCC_CPF_MI200(
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SQ_SPI_TA_TCC_CPF/MI200"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_mem_levels_HBM_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/mem_levels_HBM/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SPI_MI300X_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SPI/MI300X_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SPI_MI100(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SPI/MI100")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SPI_MI300A_A1(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir(
        "tests/workloads/ipblocks_SPI/MI300A_A1"
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_ipblocks_SPI_MI200(binary_handler_analyze_rocprof_compute):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/ipblocks_SPI/MI200")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


##################################################
##          Torch trace analysis tests          ##
##################################################


def test_analyze_torch_trace_list_operators_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--list-torch-operators",
    ])
    assert code == 0

    output = capsys.readouterr().out

    assert "PyTorch Operator Call Tree:" in output
    assert "Grouped by source location" in output
    assert "torch.nn.functional.relu" in output
    assert "torch.nn.functional.linear" in output
    assert "torch.ones_like" in output
    assert "kernel_launches:" in output
    assert "total_duration:" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_filter_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
    ])
    assert code == 0

    output = capsys.readouterr().out

    assert "Matched PyTorch Operators:" in output
    assert "relu" in output
    assert "kernel_launches:" in output
    assert "total_duration:" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_multi_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "*relu",
        "*ones_like",
    ])
    assert code == 0

    output = capsys.readouterr().out

    assert "Matched PyTorch Operators:" in output
    assert "relu" in output
    assert "ones_like" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_invalid_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "nonexistent_op",
    ])
    assert code == 0

    output = capsys.readouterr().out
    assert "No operators matched" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_hierarchy_path_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    hierarchy = "nn.Module.SimpleNet.forward/torch.nn.functional.relu/torch.relu"
    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        hierarchy,
    ])
    assert code == 0

    output = capsys.readouterr().out

    assert "Matched PyTorch Operators:" in output
    assert "torch.relu" in output
    assert "kernel_launches:" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_torch_prefix_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = test_utils.setup_workload_dir("tests/workloads/torch_trace/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "--experimental",
        "analyze",
        "--path",
        workload_dir,
        "--torch-operator",
        "torch.relu",
    ])
    assert code == 0

    output = capsys.readouterr().out

    assert "Matched PyTorch Operators:" in output
    assert "torch.relu" in output
    assert "kernel_launches:" in output

    test_utils.clean_output_dir(config["cleanup"], workload_dir)
