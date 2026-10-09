# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
from types import SimpleNamespace

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


def torch_parent_triton_child_forest():
    child = CallTreeNode(name="triton.JITFunction.matmul_kernel", backend="triton")
    child.kernels["triton_matmul_kernel"] = KernelStats(
        launches=1, total_duration_ns=40.0
    )
    parent = CallTreeNode(name="nn.Module.Linear.forward", backend="torch")
    parent.kernels["torch_gemm_kernel"] = KernelStats(
        launches=1, total_duration_ns=10.0
    )
    parent.children = [child]
    return {"1": [parent]}


def workload_with_operator_forest():
    workload = schema.Workload()
    workload.ml_api_call_trees = simple_model_forest_with_relu_and_addmm()
    workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = pd.DataFrame({
        "Kernel_Name": ["addmm_kernel", "relu_kernel"]
    })
    return workload


def apply_torch_operator_glob(pattern):
    args = argparse.Namespace(torch_operator=[pattern])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    return workload


# -- pre_processing: membw auto-run -------------------------------------------


@pytest.mark.parametrize(
    "membw_collected, expect_called",
    [
        pytest.param(True, True, id="collected_runs_analysis"),
        pytest.param(False, False, id="not_collected_skips_analysis"),
    ],
)
def test_pre_processing_membw_auto_run(membw_collected, expect_called, monkeypatch):
    """run_membw_analysis is called iff profiling config recorded membw data."""
    inst = cli_analysis.__new__(cli_analysis)
    inst._profiling_config = {"membw_analysis": membw_collected}

    workload = SimpleNamespace(
        dfs={1: pd.DataFrame()},
        sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
        raw_pmc=pd.DataFrame(),
        filter_gpu_ids=None,
        filter_dispatch_ids=None,
        membw_result=None,
    )
    inst._runs = {"/tmp/test": workload}
    inst._arch_configs = {"gfx950": SimpleNamespace(dfs_expressions={})}

    args = argparse.Namespace(
        path=[["/tmp/test"]],
        verbose=0,
        time_unit="ns",
        torch_operator=None,
        triton_operator=None,
        ml_api_operator=None,
        torch_list_ops=False,
        triton_list_ops=False,
        ml_api_list_ops=False,
    )
    inst._OmniAnalyze_Base__args = args

    membw_calls: list[tuple] = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.run_membw_analysis",
        lambda *a, **kw: membw_calls.append(a),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.pre_processing",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis.pc_sampling_only",
        lambda self: False,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis.load_pc_sampling_tool_data",
        lambda self, _path: None,
    )
    monkeypatch.setattr("utils.file_io.create_df_pmc", lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(
        "utils.file_io.create_df_kernel_top_stats",
        lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr("utils.parser.load_table_data", lambda *a, **kw: None)

    inst.pre_processing()

    assert len(membw_calls) == (1 if expect_called else 0)



# -- parse_operator_patterns (torch_operator) -------------------------------


def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = argparse.Namespace(torch_operator=["relu"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu"]}

    args = argparse.Namespace(torch_operator=["relu", "conv2d"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu", "conv2d"]}


def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = argparse.Namespace(torch_operator=["relu,conv2d"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu", "conv2d"]}


def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = argparse.Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    result = parse_operator_patterns(args, ["torch"])
    assert result == {"torch": ["relu", "conv2d", "linear"]}


def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns None."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    parse = parse_operator_patterns
    assert parse(argparse.Namespace(torch_operator=[]), ["torch"]) == {"torch": ["**"]}
    assert parse(argparse.Namespace(torch_operator=None), ["torch"]) is None
    assert parse(argparse.Namespace(), ["torch"]) is None


def test_parse_operator_patterns_generic_attr():
    """parse_operator_patterns reads the given dest attribute."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = argparse.Namespace(triton_operator=["*matmul*,*softmax*"], torch_operator=None)
    assert parse_operator_patterns(args, ["triton"]) == {
        "triton": ["*matmul*", "*softmax*"]
    }
    assert parse_operator_patterns(args, ["triton"]) != parse_operator_patterns(
        args, ["torch"]
    )
    assert parse_operator_patterns(argparse.Namespace(triton_operator=[]), ["triton"]) == {
        "triton": ["**"]
    }


def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = argparse.Namespace(torch_operator=["*"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["*"]}

    args = argparse.Namespace(torch_operator=["*,torch.relu"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["*", "torch.relu"]}


def test_operator_glob_relu_selects_relu_kernel_ids():
    workload = apply_torch_operator_glob("*relu*")
    assert workload.filter_kernel_ids == [1]


def test_operator_glob_addmm_path_selects_addmm_kernel_ids():
    workload = apply_torch_operator_glob("*/aten::addmm")
    assert workload.filter_kernel_ids == [0]


def test_operator_glob_linear_includes_descendant_addmm_ids():
    workload = apply_torch_operator_glob("*Linear.forward")
    assert workload.filter_kernel_ids == [0]


def test_list_operators_joint_backend_heading(capsys):
    cli = cli_analysis.__new__(cli_analysis)
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    cli._runs = {"/workload": workload}
    kernel_top = pd.DataFrame({
        "Kernel_Name": ["torch_gemm_kernel", "triton_matmul_kernel"]
    })
    cli.list_operators("/workload", kernel_top, ["torch", "triton"])
    captured = capsys.readouterr()
    assert "PyTorch, Triton Operator Call Tree" in captured.out


def test_handle_operator_prints_matched_subtree(capsys):
    args = argparse.Namespace(torch_operator=["*addmm*"])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    cli.handle_operator(args, workload, ["torch"])
    captured = capsys.readouterr()
    assert "Matched PyTorch Operators: *addmm*" in captured.out
    assert "aten::addmm" in captured.out
    assert workload.filter_kernel_ids == [0]


def test_apply_operator_filter_intersects_existing_kernel_ids(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.console_warning",
        lambda *argv: warnings.append(argv),
    )
    args = argparse.Namespace(torch_operator=["*relu*"], kernel=[0])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    workload.filter_kernel_ids = [0]
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    assert any(
        "No PyTorch operators matched the -k filter: [0]" in str(item)
        for item in warnings
    )
    assert workload.filter_kernel_ids == [0]


def test_apply_operator_filter_keeps_intersection():
    args = argparse.Namespace(torch_operator=["*addmm*"])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    workload.filter_kernel_ids = [0]
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    assert workload.filter_kernel_ids == [0]


def test_apply_operator_filter_kernel_ids_only_from_requested_backend():
    args = argparse.Namespace(torch_operator=["*"])
    cli = cli_analysis(args, {})
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = pd.DataFrame({
        "Kernel_Name": ["torch_gemm_kernel", "triton_matmul_kernel"]
    })
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    assert workload.filter_kernel_ids == [0]
    stored_names = []

    def collect_names(node):
        stored_names.append(node.name)
        for child in node.children:
            collect_names(child)

    for roots in workload.ml_api_call_trees.values():
        for root in roots:
            collect_names(root)
    assert "triton.JITFunction.matmul_kernel" not in stored_names


def test_list_operators_omits_other_backend_kernel_lines(capsys):
    cli = cli_analysis.__new__(cli_analysis)
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    cli._runs = {"/workload": workload}
    kernel_top = pd.DataFrame({
        "Kernel_Name": ["torch_gemm_kernel", "triton_matmul_kernel"]
    })
    cli.list_operators("/workload", kernel_top, ["triton"])
    captured = capsys.readouterr()
    assert "triton_matmul_kernel" in captured.out
    assert "torch_gemm_kernel" not in captured.out
