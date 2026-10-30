# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
import functools
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_api_tracing.call_tree import iter_call_tree_paths
from ml_api_tracing.models import CallTreeNode, KernelStats
from rocprof_compute_analyze.analysis_cli import (
    _intersect_with_user_dispatch_filter,
    cli_analysis,
    parse_operator_patterns,
)
from utils import schema

ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
RELU_PMC_GROUP_KEY = (2, "relu_kernel", 0, 0, 0)


def simple_model_forest_with_relu_and_addmm():
    """SimpleModel.forward with Linear/addmm and a relu sibling."""
    addmm = CallTreeNode(name="aten::addmm", backend="torch")
    addmm.kernels["addmm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=50.0,
        pmc_group_keys=[ADDMM_PMC_GROUP_KEY],
    )
    relu = CallTreeNode(name="aten::relu", backend="torch")
    relu.kernels["relu_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[RELU_PMC_GROUP_KEY],
    )
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
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[(3, "torch_gemm_kernel", 0, 0, 0)],
    )
    parent.children = [child]
    return {"1": [parent]}


def workload_with_operator_forest():
    workload = schema.Workload()
    workload.ml_api_call_trees = simple_model_forest_with_relu_and_addmm()
    return workload


def apply_torch_operator_glob(pattern):
    args = argparse.Namespace(torch_operator=[pattern])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    pmc_group_keys = cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    return pmc_group_keys, workload


def forest_node_names(forest):
    return [
        node.name
        for roots in forest.values()
        for node, _path in iter_call_tree_paths(roots)
    ]


def append_args_and_exit(errors, *argv, **_kwargs):
    """Stand-in for console_error: record the call, then exit like it does."""
    errors.append(argv)
    raise SystemExit(1)


def cli_with_two_operator_workloads(args):
    inst = cli_analysis.__new__(cli_analysis)
    inst._profiling_config = {}
    inst._output = None
    inst._runs = {
        "/w1": workload_with_operator_forest(),
        "/w2": workload_with_operator_forest(),
    }
    for workload in inst._runs.values():
        workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx950"}])
    inst._arch_configs = {"gfx950": SimpleNamespace(dfs_expressions={})}
    inst._OmniAnalyze_Base__args = args
    return inst


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

    args = argparse.Namespace(torch_operator=["relu"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu"]}

    args = argparse.Namespace(torch_operator=["relu", "conv2d"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu", "conv2d"]}


def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""

    args = argparse.Namespace(torch_operator=["relu,conv2d"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["relu", "conv2d"]}


def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""

    args = argparse.Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    result = parse_operator_patterns(args, ["torch"])
    assert result == {"torch": ["relu", "conv2d", "linear"]}


def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns None."""

    parse = parse_operator_patterns
    assert parse(argparse.Namespace(torch_operator=[]), ["torch"]) == {"torch": ["**"]}
    assert parse(argparse.Namespace(torch_operator=None), ["torch"]) is None
    assert parse(argparse.Namespace(), ["torch"]) is None


def test_parse_operator_patterns_generic_attr():
    """parse_operator_patterns reads the given dest attribute."""

    args = argparse.Namespace(
        triton_operator=["*matmul*,*softmax*"], torch_operator=None
    )
    assert parse_operator_patterns(args, ["triton"]) == {
        "triton": ["*matmul*", "*softmax*"]
    }
    assert parse_operator_patterns(args, ["triton"]) != parse_operator_patterns(
        args, ["torch"]
    )
    assert parse_operator_patterns(
        argparse.Namespace(triton_operator=[]), ["triton"]
    ) == {"triton": ["**"]}


def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""

    args = argparse.Namespace(torch_operator=["*"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["*"]}

    args = argparse.Namespace(torch_operator=["*,torch.relu"])
    assert parse_operator_patterns(args, ["torch"]) == {"torch": ["*", "torch.relu"]}


def test_operator_glob_relu_returns_relu_keys():
    pmc_group_keys, workload = apply_torch_operator_glob("*relu*")
    assert pmc_group_keys == [RELU_PMC_GROUP_KEY]
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "aten::relu",
    ]


def test_operator_glob_addmm_path_returns_addmm_keys():
    pmc_group_keys, workload = apply_torch_operator_glob("*/aten::addmm")
    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


def test_operator_glob_linear_includes_descendant_addmm_keys():
    pmc_group_keys, workload = apply_torch_operator_glob("*Linear.forward")
    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


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


def test_apply_operator_filter_missed_glob_errors(monkeypatch):
    errors = []

    def record_error(*argv, **_kwargs):
        errors.append(argv)
        raise SystemExit(1)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.console_error",
        record_error,
    )
    args = argparse.Namespace(torch_operator=["nomatch"])
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    with pytest.raises(SystemExit) as excinfo:
        cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    assert excinfo.value.code == 1
    assert any(
        "No PyTorch operators matched the pattern(s): nomatch" in str(item)
        for item in errors
    )
    assert "aten::addmm" in forest_node_names(workload.ml_api_call_trees)


def test_dispatch_filter_keeps_only_operator_dispatches():
    assert _intersect_with_user_dispatch_filter([1, 2, 3], ["2", "9"], "PyTorch") == [2]


def test_dispatch_filter_outside_operator_dispatches_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.console_error",
        functools.partial(append_args_and_exit, errors),
    )
    with pytest.raises(SystemExit):
        _intersect_with_user_dispatch_filter([1, 2, 3], ["9"], "PyTorch")
    assert "No PyTorch dispatches matched the -d filter" in str(errors[-1])


def test_apply_operator_filter_keeps_nodes_from_requested_backend():
    args = argparse.Namespace(torch_operator=["*"])
    cli = cli_analysis(args, {})
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    names = forest_node_names(workload.ml_api_call_trees)
    assert "nn.Module.Linear.forward" in names
    assert "triton.JITFunction.matmul_kernel" not in names


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


def test_pre_processing_lists_operators_for_every_path(monkeypatch):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        verbose=0,
        time_unit="ns",
        torch_operator=None,
        triton_operator=None,
        list_torch_operators=True,
        list_triton_operators=False,
    )
    inst = cli_with_two_operator_workloads(args)
    listed_paths = []
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
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.process_ml_api_trace_output",
        lambda workload, workload_dir: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis.list_operators",
        lambda self, workload_path, kernel_top, backends: listed_paths.append(
            workload_path
        ),
    )
    with pytest.raises(SystemExit) as excinfo:
        inst.pre_processing()
    assert excinfo.value.code == 0
    assert listed_paths == ["/w1", "/w2"]


def test_run_analysis_prints_matched_tree_for_every_path(monkeypatch):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        torch_operator=["*addmm*"],
        triton_operator=None,
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(args)
    handled_paths = []

    def capture_handle(self, handle_args, workload, backends):
        for workload_path, run in inst._runs.items():
            if run is workload:
                handled_paths.append(workload_path)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.run_analysis",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis.handle_operator",
        capture_handle,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda *a, **kw: None,
    )
    inst.run_analysis()
    assert handled_paths == ["/w1", "/w2"]
