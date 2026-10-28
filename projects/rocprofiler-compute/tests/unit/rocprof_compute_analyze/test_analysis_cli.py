# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
import functools
from types import SimpleNamespace

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import (
    _intersect_with_user_dispatch_filter,
    cli_analysis,
)
from utils import schema
from utils.ml_api_selectors import resolve_operator_patterns
from utils.utils_analysis import CallTreeNode, KernelStats, iter_call_tree_paths

ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
RELU_PMC_GROUP_KEY = (2, "relu_kernel", 0, 0, 0)
RMSNORM_PMC_GROUP_KEY = (4, "rmsnorm_kernel", 0, 0, 0)


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


def user_region_with_nested_addmm_and_outside_relu():
    """User region1 wraps addmm; relu is a sibling root outside the region."""
    addmm = CallTreeNode(name="aten::addmm", backend="torch")
    addmm.kernels["addmm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=50.0,
        pmc_group_keys=[ADDMM_PMC_GROUP_KEY],
    )
    region1 = CallTreeNode(name="region1", backend="user")
    region1.children = [addmm]
    relu = CallTreeNode(name="aten::relu", backend="torch")
    relu.kernels["relu_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[RELU_PMC_GROUP_KEY],
    )
    return {"1": [region1, relu]}


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


def triton_rmsnorm_forest():
    rmsnorm = CallTreeNode(name="triton.JITFunction.rmsnorm_kernel", backend="triton")
    rmsnorm.kernels["rmsnorm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=20.0,
        pmc_group_keys=[RMSNORM_PMC_GROUP_KEY],
    )
    return {"1": [rmsnorm]}


def workload_with_operator_forest():
    workload = schema.Workload()
    workload.ml_api_call_trees = simple_model_forest_with_relu_and_addmm()
    return workload


def stub_trace_ml_api(monkeypatch, traced_by_path=None):
    """Make load_profiling_config return trace_ml_api for each workload path."""
    configs = traced_by_path or {}

    def load_profiling_config(workload_path):
        return {"trace_ml_api": list(configs.get(workload_path, ["torch"]))}

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.file_io.load_profiling_config",
        load_profiling_config,
    )


def apply_torch_operator_glob(pattern, monkeypatch):
    stub_trace_ml_api(monkeypatch)
    args = argparse.Namespace(
        operator_patterns=resolve_operator_patterns([f"torch:{pattern}"])
    )
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    pmc_group_keys, _ = cli.apply_operator_filter(args, workload, "/workload")
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
        operator_patterns=None,
        list_operators=None,
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


def test_operator_glob_relu_returns_relu_keys(monkeypatch):
    pmc_group_keys, workload = apply_torch_operator_glob("*relu*", monkeypatch)
    assert pmc_group_keys == [RELU_PMC_GROUP_KEY]
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "aten::relu",
    ]


def test_operator_glob_addmm_path_returns_addmm_keys(monkeypatch):
    pmc_group_keys, workload = apply_torch_operator_glob("*/aten::addmm", monkeypatch)
    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


def test_operator_glob_linear_includes_descendant_addmm_keys(monkeypatch):
    pmc_group_keys, workload = apply_torch_operator_glob("*Linear.forward", monkeypatch)
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


def test_handle_operator_prints_matched_subtree(capsys, monkeypatch):
    stub_trace_ml_api(monkeypatch)
    args = argparse.Namespace(
        operator_patterns=resolve_operator_patterns(["torch:*addmm*"])
    )
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload")
    cli.handle_operator(args, workload, "/workload")
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
    stub_trace_ml_api(monkeypatch)
    args = argparse.Namespace(
        operator_patterns=resolve_operator_patterns(["torch:nomatch"])
    )
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    with pytest.raises(SystemExit) as excinfo:
        cli.apply_operator_filter(args, workload, "/workload")
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


def test_apply_operator_filter_keeps_nodes_from_requested_backend(monkeypatch):
    stub_trace_ml_api(monkeypatch)
    args = argparse.Namespace(operator_patterns=resolve_operator_patterns(["torch:*"]))
    cli = cli_analysis(args, {})
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    cli.apply_operator_filter(args, workload, "/workload")
    names = forest_node_names(workload.ml_api_call_trees)
    assert "nn.Module.Linear.forward" in names
    assert "triton.JITFunction.matmul_kernel" not in names


def test_apply_operator_filter_user_region_keeps_nested_work_and_unions_relu(
    monkeypatch,
):
    stub_trace_ml_api(monkeypatch)
    workload = schema.Workload()
    workload.ml_api_call_trees = user_region_with_nested_addmm_and_outside_relu()
    args = argparse.Namespace(
        operator_patterns=resolve_operator_patterns(["user:region1"])
    )
    region_keys, _ = cli_analysis(args, {}).apply_operator_filter(
        args, workload, "/workload"
    )
    assert region_keys == [ADDMM_PMC_GROUP_KEY]
    region_names = forest_node_names(workload.ml_api_call_trees)
    assert region_names == ["region1", "aten::addmm"]

    union_workload = schema.Workload()
    union_workload.ml_api_call_trees = user_region_with_nested_addmm_and_outside_relu()
    union_args = argparse.Namespace(
        operator_patterns=resolve_operator_patterns(["user:region1", "torch:*relu*"])
    )
    union_keys, _ = cli_analysis(union_args, {}).apply_operator_filter(
        union_args, union_workload, "/workload"
    )
    assert union_keys == [ADDMM_PMC_GROUP_KEY, RELU_PMC_GROUP_KEY]
    assert forest_node_names(union_workload.ml_api_call_trees) == [
        "region1",
        "aten::addmm",
        "aten::relu",
    ]


def test_apply_operator_filter_unscoped_mixed_paths(monkeypatch):
    """Bare --operator on sn_torch then tf_triton binds each path's yaml."""
    stub_trace_ml_api(
        monkeypatch,
        {"/sn_torch": ["torch"], "/tf_triton": ["triton"]},
    )
    args = argparse.Namespace(operator_patterns=resolve_operator_patterns([]))
    cli = cli_analysis(args, {})
    torch_workload = workload_with_operator_forest()
    triton_workload = schema.Workload()
    triton_workload.ml_api_call_trees = triton_rmsnorm_forest()
    torch_keys, _ = cli.apply_operator_filter(args, torch_workload, "/sn_torch")
    triton_keys, _ = cli.apply_operator_filter(args, triton_workload, "/tf_triton")
    assert ADDMM_PMC_GROUP_KEY in torch_keys
    assert triton_keys == [RMSNORM_PMC_GROUP_KEY]


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


def test_list_operators_user_keeps_nested_torch_and_drops_outside_relu(capsys):
    cli = cli_analysis.__new__(cli_analysis)
    workload = schema.Workload()
    workload.ml_api_call_trees = user_region_with_nested_addmm_and_outside_relu()
    cli._runs = {"/workload": workload}
    kernel_top = pd.DataFrame({"Kernel_Name": ["addmm_kernel", "relu_kernel"]})
    cli.list_operators("/workload", kernel_top, ["user"])
    captured = capsys.readouterr()
    assert "User Operator Call Tree:" in captured.out
    assert "region1" in captured.out
    assert "aten::addmm" in captured.out
    assert "aten::relu" not in captured.out


def test_pre_processing_lists_operators_for_every_path(monkeypatch):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        verbose=0,
        time_unit="ns",
        operator_patterns=None,
        list_operators=["torch"],
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
        operator_patterns=resolve_operator_patterns(["torch:*addmm*"]),
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(args)
    handled_paths = []

    def capture_handle(self, handle_args, workload, workload_path):
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
