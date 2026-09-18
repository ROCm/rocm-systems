# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

from argparse import Namespace

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import cli_analysis
from utils import parser, schema
from utils.utils_analysis import CallTreeNode, KernelStats


def simple_model_forest_with_relu_and_addmm():
    """SimpleModel.forward with Linear/addmm and a relu sibling."""
    addmm = CallTreeNode(name="aten::addmm", backend="torch")
    addmm.kernels["addmm_kernel"] = KernelStats(launches=1, total_duration_ns=50.0)
    relu = CallTreeNode(name="aten::relu", backend="torch")
    relu.kernels["relu_kernel"] = KernelStats(launches=1, total_duration_ns=10.0)
    linear = CallTreeNode(name="nn.Module.Linear.forward", backend="torch")
    linear.children = [addmm]
    simple = CallTreeNode(name="nn.Module.SimpleModel.forward", backend="torch")
    simple.children = [linear, relu]
    return {"1": [simple]}


def workload_with_operator_forest():
    workload = schema.Workload()
    workload.ml_api_call_trees = simple_model_forest_with_relu_and_addmm()
    workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = pd.DataFrame({
        "Kernel_Name": ["addmm_kernel", "relu_kernel"]
    })
    return workload


def apply_torch_operator_glob(pattern):
    args = Namespace(torch_operator=[pattern])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload", "torch")
    return workload


# -- parse_operator_patterns (torch_operator) -------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu"]

    args = Namespace(torch_operator=["relu", "conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["relu,conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    result = parse_operator_patterns(args, "torch_operator")
    assert result == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    parse = parse_operator_patterns
    assert parse(Namespace(torch_operator=[]), "torch_operator") == ["**"]
    assert parse(Namespace(torch_operator=None), "torch_operator") == []
    assert parse(Namespace(), "torch_operator") == []


# -- parse_operator_patterns / triton backend selection ---------------------


@pytest.mark.torch_ops
def test_parse_operator_patterns_generic_attr():
    """parse_operator_patterns reads the given dest attribute."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(triton_operator=["*matmul*,*softmax*"], torch_operator=None)
    assert parse_operator_patterns(args, "triton_operator") == [
        "*matmul*",
        "*softmax*",
    ]
    assert parse_operator_patterns(args, "triton_operator") != parse_operator_patterns(
        args, "torch_operator"
    )
    assert parse_operator_patterns(
        Namespace(triton_operator=[]), "triton_operator"
    ) == ["**"]


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["*"])
    assert parse_operator_patterns(args, "torch_operator") == ["*"]

    args = Namespace(torch_operator=["*,torch.relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["*", "torch.relu"]


@pytest.mark.torch_ops
def test_operator_glob_relu_selects_relu_kernel_ids():
    workload = apply_torch_operator_glob("*relu*")
    assert workload.filter_kernel_ids == [1]


@pytest.mark.torch_ops
def test_operator_glob_addmm_path_selects_addmm_kernel_ids():
    workload = apply_torch_operator_glob("*/aten::addmm")
    assert workload.filter_kernel_ids == [0]


@pytest.mark.torch_ops
def test_operator_glob_linear_includes_descendant_addmm_ids():
    workload = apply_torch_operator_glob("*Linear.forward")
    assert workload.filter_kernel_ids == [0]
