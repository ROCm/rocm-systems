# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

##################################################
##          Generated tests                     ##
##################################################

import os

import common
import pytest

config = {}
config["cleanup"] = True if "PYTEST_XDIST_WORKER_COUNT" in os.environ else False

# MI300A_A1 and MI300X_A1 share the same workload set and expected exit codes.
# Note: dispatch_6_8, dispatch_7, kernel_inv_int, kernel_inv_str were profiled
# with rocprofv2, which collects all dispatches regardless of the filter applied
# at profile time. The analyze command therefore succeeds (exit 0) for these
# targets, unlike MI100/MI200 where rocprofv1 applied the filter at collection
# time and produced no joinable data (exit 1).
MI300_WORKLOADS = [
    ("device_filter", 0),
    ("device_inv_int", 0),
    ("dispatch_0", 0),
    ("dispatch_0_1", 0),
    ("dispatch_2", 0),
    ("dispatch_6_8", 0),
    ("dispatch_7", 0),
    ("dispatch_inv", 0),
    ("ipblocks_CPC", 0),
    ("ipblocks_CPF", 0),
    ("ipblocks_SPI", 0),
    ("ipblocks_SQ", 0),
    ("ipblocks_SQC", 0),
    ("ipblocks_SQ_CPC", 0),
    ("ipblocks_SQ_SPI", 0),
    ("ipblocks_SQ_SPI_TA_TCC_CPF", 0),
    ("ipblocks_SQ_SQC_TCP_CPC", 0),
    ("ipblocks_SQ_TA", 0),
    ("ipblocks_TA", 0),
    ("ipblocks_TCC", 0),
    ("ipblocks_TCP", 0),
    ("ipblocks_TD", 0),
    ("join_type_grid", 0),
    ("join_type_kernel", 0),
    ("kernel", 0),
    ("kernel_inv_int", 0),
    ("kernel_inv_str", 0),
    ("kernel_substr", 0),
    ("no_roof", 0),
    ("path", 0),
]

WORKLOADS_BY_ARCH = {
    "MI100": [
        ("device_filter", 0),
        ("device_inv_int", 0),
        ("dispatch_0", 0),
        ("dispatch_0_1", 0),
        ("dispatch_2", 0),
        ("dispatch_6_8", 1),
        ("dispatch_7", 1),
        ("dispatch_inv", 0),
        ("ipblocks_CPC", 0),
        ("ipblocks_CPF", 0),
        ("ipblocks_SPI", 0),
        ("ipblocks_SQ", 0),
        ("ipblocks_SQC", 0),
        ("ipblocks_SQ_CPC", 0),
        ("ipblocks_SQ_SPI", 0),
        ("ipblocks_SQ_SPI_TA_TCC_CPF", 0),
        ("ipblocks_SQ_SQC_TCP_CPC", 0),
        ("ipblocks_SQ_TA", 0),
        ("ipblocks_TA", 0),
        ("ipblocks_TCC", 0),
        ("ipblocks_TCP", 0),
        ("ipblocks_TD", 0),
        ("join_type_grid", 0),
        ("join_type_kernel", 0),
        ("kernel", 0),
        ("kernel_inv_int", 1),
        ("kernel_inv_str", 1),
        ("kernel_substr", 0),
        ("no_roof", 0),
        ("path", 0),
        ("vcopy", 0),
    ],
    "MI200": [
        ("device_filter", 0),
        ("device_inv_int", 0),
        ("dispatch_0", 0),
        ("dispatch_0_1", 0),
        ("dispatch_2", 0),
        ("dispatch_6_8", 1),
        ("dispatch_7", 1),
        ("dispatch_inv", 0),
        ("ipblocks_CPC", 0),
        ("ipblocks_CPF", 0),
        ("ipblocks_SPI", 0),
        ("ipblocks_SQ", 0),
        ("ipblocks_SQC", 0),
        ("ipblocks_SQ_CPC", 0),
        ("ipblocks_SQ_SPI", 0),
        ("ipblocks_SQ_SPI_TA_TCC_CPF", 0),
        ("ipblocks_SQ_SQC_TCP_CPC", 0),
        ("ipblocks_SQ_TA", 0),
        ("ipblocks_TA", 0),
        ("ipblocks_TCC", 0),
        ("ipblocks_TCP", 0),
        ("ipblocks_TD", 0),
        ("join_type_grid", 0),
        ("join_type_kernel", 0),
        ("kernel", 0),
        ("kernel_inv_int", 1),
        ("kernel_inv_str", 1),
        ("kernel_names", 0),
        ("kernel_substr", 0),
        ("mem_levels_HBM", 0),
        ("mem_levels_HBM_LDS", 0),
        ("mem_levels_L2", 0),
        ("mem_levels_L2_vL1d_LDS", 0),
        ("mem_levels_LDS", 0),
        ("mem_levels_vL1D", 0),
        ("mem_levels_vL1d_LDS", 0),
        ("no_roof", 0),
        ("path", 0),
        ("sort_dispatches", 0),
        ("sort_kernels", 0),
        ("vcopy", 0),
    ],
    "MI300A_A1": MI300_WORKLOADS,
    "MI300X_A1": MI300_WORKLOADS,
    "RDNA35_HALO": [
        ("dispatch_0", 0),
        ("ipblocks_CU", 0),
        ("kernel", 0),
        ("no_roof", 0),
        ("path", 0),
        ("vcopy", 0),
    ],
}


@pytest.mark.parametrize(
    "workload_type,expected_code",
    WORKLOADS_BY_ARCH["MI100"],
    ids=[f"workload={w}-exit={c}" for w, c in WORKLOADS_BY_ARCH["MI100"]],
)
def test_analyze_MI100(
    binary_handler_analyze_rocprof_compute, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/MI100", param_id=workload_type
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == expected_code

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "workload_type,expected_code",
    WORKLOADS_BY_ARCH["MI200"],
    ids=[f"workload={w}-exit={c}" for w, c in WORKLOADS_BY_ARCH["MI200"]],
)
def test_analyze_MI200(
    binary_handler_analyze_rocprof_compute, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/MI200", param_id=workload_type
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == expected_code

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "workload_type,expected_code",
    WORKLOADS_BY_ARCH["MI300A_A1"],
    ids=[f"workload={w}-exit={c}" for w, c in WORKLOADS_BY_ARCH["MI300A_A1"]],
)
def test_analyze_MI300A_A1(
    binary_handler_analyze_rocprof_compute, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/MI300A_A1", param_id=workload_type
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == expected_code

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "workload_type,expected_code",
    WORKLOADS_BY_ARCH["MI300X_A1"],
    ids=[f"workload={w}-exit={c}" for w, c in WORKLOADS_BY_ARCH["MI300X_A1"]],
)
def test_analyze_MI300X_A1(
    binary_handler_analyze_rocprof_compute, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/MI300X_A1", param_id=workload_type
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == expected_code

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_MI350(binary_handler_analyze_rocprof_compute):
    workload_dir = common.setup_workload_dir("tests/workloads/no_roof/MI350")

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


@pytest.mark.parametrize(
    "workload_type,expected_code",
    WORKLOADS_BY_ARCH["RDNA35_HALO"],
    ids=[f"workload={w}-exit={c}" for w, c in WORKLOADS_BY_ARCH["RDNA35_HALO"]],
)
def test_analyze_RDNA35_HALO(
    binary_handler_analyze_rocprof_compute, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/RDNA35_HALO", param_id=workload_type
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == expected_code

    common.clean_output_dir(config["cleanup"], workload_dir)


##################################################
##          Torch trace analysis tests          ##
##################################################


def test_analyze_torch_trace_list_operators_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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
    assert "dispatches:" in output
    assert "total:" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_filter_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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
    assert "dispatches:" in output
    assert "total:" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_multi_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_invalid_operator_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_hierarchy_path_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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
    assert "dispatches:" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_torch_trace_torch_prefix_MI350(
    binary_handler_analyze_rocprof_compute, capsys
):
    workload_dir = common.setup_workload_dir("tests/workloads/torch_trace/MI350")

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
    assert "dispatches:" in output

    common.clean_output_dir(config["cleanup"], workload_dir)
