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

# 30 workloads common to MI100, MI200, MI300A_A1, MI300X_A1.
CDNA_WORKLOADS = [
    "device_filter",
    "device_inv_int",
    "dispatch_0",
    "dispatch_0_1",
    "dispatch_2",
    "dispatch_6_8",
    "dispatch_7",
    "dispatch_inv",
    "ipblocks_CPC",
    "ipblocks_CPF",
    "ipblocks_SPI",
    "ipblocks_SQ",
    "ipblocks_SQC",
    "ipblocks_SQ_CPC",
    "ipblocks_SQ_SPI",
    "ipblocks_SQ_SPI_TA_TCC_CPF",
    "ipblocks_SQ_SQC_TCP_CPC",
    "ipblocks_SQ_TA",
    "ipblocks_TA",
    "ipblocks_TCC",
    "ipblocks_TCP",
    "ipblocks_TD",
    "join_type_grid",
    "join_type_kernel",
    "kernel",
    "kernel_inv_int",
    "kernel_inv_str",
    "kernel_substr",
    "no_roof",
    "path",
]

WORKLOADS_BY_ARCH = {
    "MI100": CDNA_WORKLOADS + ["vcopy"],
    "MI200": CDNA_WORKLOADS
    + [
        "kernel_names",
        "mem_levels_HBM",
        "mem_levels_HBM_LDS",
        "mem_levels_L2",
        "mem_levels_L2_vL1d_LDS",
        "mem_levels_LDS",
        "mem_levels_vL1D",
        "mem_levels_vL1d_LDS",
        "sort_dispatches",
        "sort_kernels",
        "vcopy",
    ],
    "MI300A_A1": CDNA_WORKLOADS,
    "MI300X_A1": CDNA_WORKLOADS,
    "MI350": ["no_roof", "vcopy", "vcopy_iteration_multiplexing"],
    "RDNA35_HALO": ["dispatch_0", "ipblocks_CU", "kernel", "no_roof", "path", "vcopy"],
}

# All workloads exit 0 except these.
# dispatch_6_8 and dispatch_7 exit 1 on MI100/MI200: rocprofiler-sdk applies
# the dispatch filter at collection time, so no pmc data is written
EXIT_CODES = {
    ("MI100", "dispatch_6_8"): 1,
    ("MI100", "dispatch_7"): 1,
    ("MI200", "dispatch_6_8"): 1,
    ("MI200", "dispatch_7"): 1,
}

ANALYZE_PARAMS = [
    pytest.param(
        arch,
        workload,
        EXIT_CODES.get((arch, workload), 0),
        id=f"arch={arch}-workload={workload}-exit={EXIT_CODES.get((arch, workload), 0)}",  # noqa: E501
    )
    for arch, workloads in WORKLOADS_BY_ARCH.items()
    for workload in workloads
]


@pytest.mark.parametrize("arch,workload_type,expected_code", ANALYZE_PARAMS)
def test_analyze_workload(
    binary_handler_analyze_rocprof_compute, arch, workload_type, expected_code
):
    workload_dir = common.setup_workload_dir(
        f"tests/workloads/{workload_type}/{arch}",
        param_id=f"{arch}_{workload_type}",
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
