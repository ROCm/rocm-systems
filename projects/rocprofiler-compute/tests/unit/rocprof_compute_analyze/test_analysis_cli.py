# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import (
    cli_analysis,
    parse_operator_patterns,
)
from rocprof_compute_analyze.output_location import workload_folder_name
from utils import parser, schema
from utils.terminal_selection import TerminalSelection
from utils.utils_analysis import CallTreeNode, KernelStats, iter_call_tree_paths

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
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    return workload


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


def make_workload_folders(run_folder, workload_paths):
    """Create the workload folders that the layout writer creates before the reports."""
    for position, workload_path in enumerate(workload_paths, start=1):
        (run_folder / workload_folder_name(position, Path(workload_path))).mkdir()


def cli_with_two_operator_workloads(args):
    inst = cli_analysis.__new__(cli_analysis)
    inst._profiling_config = {}
    inst._runs = {
        "/w1": workload_with_operator_forest(),
        "/w2": workload_with_operator_forest(),
    }
    for workload in inst._runs.values():
        workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx950"}])
    inst._arch_configs = {"gfx950": SimpleNamespace(dfs_expressions={})}
    inst._OmniAnalyze_Base__args = args
    inst._terminal_selection = TerminalSelection()
    return inst


def listing_instance(monkeypatch, args):
    """A cli_analysis whose listing reads no profile data from disk."""
    inst = cli_with_two_operator_workloads(args)
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.pre_processing",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis._listing_counters",
        lambda self, _path: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "utils.file_io.create_df_kernel_top_stats",
        lambda **_kwargs: (
            pd.DataFrame({"Kernel_Name": ["addmm_kernel", "relu_kernel"]}),
            pd.DataFrame(),
        ),
    )
    return inst


# -- pre_processing: membw auto-run -------------------------------------------


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
    workload = apply_torch_operator_glob("*relu*")
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "aten::relu",
    ]


def test_operator_glob_addmm_path_returns_addmm_keys():
    workload = apply_torch_operator_glob("*/aten::addmm")
    assert forest_node_names(workload.ml_api_call_trees) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


def test_operator_glob_linear_includes_descendant_addmm_keys():
    workload = apply_torch_operator_glob("*Linear.forward")
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
    args = argparse.Namespace(torch_operator=["*addmm*"], decimal=2)
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    kernel_top = pd.DataFrame({"Kernel_Name": ["addmm_kernel", "relu_kernel"]})
    cli.handle_operator(
        args, workload, ["torch"], ["addmm_kernel", "relu_kernel"], kernel_top
    )
    captured = capsys.readouterr()
    assert "Matched PyTorch Operators: *addmm*" in captured.out
    assert "aten::addmm" in captured.out
    assert "Kernels launched by the matched PyTorch operators" in captured.out
    assert "addmm_kernel" in captured.out
    assert "relu_kernel" not in captured.out


def test_handle_operator_ids_follow_the_full_ranking_under_k(capsys):
    """The ids are the full ranking's, so -k keeps the id of the kernel it selects.

    The report's Top Kernels holds only relu_kernel, the -k 1 selection, so
    addmm_kernel has no row here and -k 1 is the id the tree shows for relu.
    """
    args = argparse.Namespace(torch_operator=["*addmm*"], decimal=2)
    cli = cli_analysis(args, {})
    workload = workload_with_operator_forest()
    cli.apply_operator_filter(args, workload, "/workload", ["torch"])
    report_kernels = pd.DataFrame({"Kernel_Name": ["relu_kernel"]})
    cli.handle_operator(
        args,
        workload,
        ["torch"],
        ["relu_kernel", "addmm_kernel"],
        report_kernels,
    )
    captured = capsys.readouterr()
    assert "addmm_kernel (id 1)" in captured.out
    assert "No kernels launched by the matched operators are in this run." in (
        captured.out
    )
    assert "left out by -k: kernel id(s) 1" in captured.out


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


def test_listing_lists_operators_for_every_path(monkeypatch):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        verbose=0,
        time_unit="ns",
        torch_operator=None,
        triton_operator=None,
        list_stats=False,
        list_torch_operators=True,
        list_triton_operators=False,
    )
    inst = listing_instance(monkeypatch, args)
    listed_paths = []
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


def test_listing_without_operators_builds_no_operator_trees(monkeypatch):
    """--list-stats reads the kernels only; the ML API trace is not processed."""
    args = argparse.Namespace(
        path=[["/w1"]],
        verbose=0,
        time_unit="ns",
        torch_operator=None,
        triton_operator=None,
        list_stats=True,
        list_torch_operators=False,
        list_triton_operators=False,
    )
    inst = listing_instance(monkeypatch, args)
    traced = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.process_ml_api_trace_output",
        lambda workload, workload_dir: traced.append(workload_dir),
    )

    inst.pre_processing()

    assert traced == []
    assert (
        "addmm_kernel"
        in inst._runs["/w1"].dfs[parser.PMC_KERNEL_TOP_TABLE_ID]["Kernel_Name"].tolist()
    )


def test_operator_filter_narrows_only_the_operator_forest(monkeypatch):
    """The operator filter narrows the forest the report shows and leaves the
    Top Kernels frame, which the analysis DB stores, as it was."""
    args = argparse.Namespace(
        path=[["/w1"]],
        verbose=0,
        time_unit="ns",
        torch_operator=["*addmm*"],
        triton_operator=None,
        list_stats=False,
        list_torch_operators=False,
        list_triton_operators=False,
        filter_metrics=None,
    )
    inst = cli_with_two_operator_workloads(args)
    inst._runs = {"/w1": workload_with_operator_forest()}
    inst._runs["/w1"].sys_info = pd.DataFrame([{"gpu_arch": "gfx950"}])
    kernel_top = pd.DataFrame({"Kernel_Name": ["addmm_kernel", "relu_kernel"]})
    inst._runs["/w1"].dfs = {parser.PMC_KERNEL_TOP_TABLE_ID: kernel_top.copy()}
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.pre_processing",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.build_terminal_selection",
        lambda *_args: TerminalSelection(),
    )

    inst.pre_processing()

    names = forest_node_names(inst._runs["/w1"].ml_api_call_trees)
    assert names == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]
    pd.testing.assert_frame_equal(
        inst._runs["/w1"].dfs[parser.PMC_KERNEL_TOP_TABLE_ID], kernel_top
    )


def test_write_outputs_prints_matched_tree_for_every_path(monkeypatch, tmp_path):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        torch_operator=["*addmm*"],
        triton_operator=None,
        list_stats=False,
        decimal=2,
    )
    inst = cli_with_two_operator_workloads(args)
    kernel_top = pd.DataFrame({"Kernel_Name": ["addmm_kernel", "relu_kernel"]})
    inst._report_runs = {
        "/w1": schema.Workload(dfs={1: kernel_top}),
        "/w2": schema.Workload(dfs={1: kernel_top}),
    }
    inst._kernel_rankings = {
        "/w1": ["addmm_kernel", "relu_kernel"],
        "/w2": ["addmm_kernel", "relu_kernel"],
    }
    handled_paths = []

    def capture_handle(self, handle_args, workload, backends, ranking, top):
        for workload_path, run in inst._runs.items():
            if run is workload:
                handled_paths.append(workload_path)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.cli_analysis.handle_operator",
        capture_handle,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda *a, **kw: None,
    )
    make_workload_folders(tmp_path, ["/w1", "/w2"])

    inst.write_outputs(tmp_path)

    # The root report, each workload's own report, then the terminal
    assert handled_paths == ["/w1", "/w2", "/w1", "/w2", "/w1", "/w2"]


def test_write_outputs_writes_full_report_and_prints_the_terminal_selection(
    monkeypatch, tmp_path
):
    """report.txt gets every panel; stdout gets the terminal selection and the
    roofline HTML path."""
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        torch_operator=None,
        triton_operator=None,
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(args)
    inst._report_runs = {"/w1": schema.Workload(), "/w2": schema.Workload()}
    inst._terminal_selection = TerminalSelection(metric_ids=frozenset({"7"}))
    show_all_calls = []
    path_outputs = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda _args, _runs, _config, output, _profiling, **kwargs: (
            show_all_calls.append((output, kwargs.get("terminal_selection")))
        ),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.print_roofline_html_path",
        lambda _runs, output: path_outputs.append(output),
    )

    make_workload_folders(tmp_path, ["/w1", "/w2"])

    inst.write_outputs(tmp_path)

    *report_calls, (stdout_output, stdout_selection) = show_all_calls
    # Each workload's report comes before the root comparison, which narrows the frames
    assert [report_output.name for report_output, _selection in report_calls] == [
        str(tmp_path / "1-w1" / "report.txt"),
        str(tmp_path / "2-w2" / "report.txt"),
        str(tmp_path / "report.txt"),
    ]
    assert all(selection is None for _output, selection in report_calls)
    assert stdout_output is sys.stdout
    assert stdout_selection is inst._terminal_selection
    assert path_outputs == [sys.stdout]


def test_write_outputs_writes_only_the_root_report_for_one_workload(
    monkeypatch, tmp_path
):
    """A single -p gets report.txt at the root and no per-workload report."""
    args = argparse.Namespace(
        path=[["/w1"]],
        torch_operator=None,
        triton_operator=None,
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(args)
    inst._report_runs = {"/w1": schema.Workload()}
    report_names = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda _args, _runs, _config, output, _profiling, **kwargs: report_names.append(
            output.name
        ),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.print_roofline_html_path",
        lambda _runs, _output: None,
    )

    inst.write_outputs(tmp_path)

    # The terminal view is printed to stdout, so only the root report is a file.
    assert report_names == [str(tmp_path / "report.txt"), sys.stdout.name]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.txt"]


def test_write_outputs_saves_each_traced_workloads_ml_api_trace(monkeypatch, tmp_path):
    """Only a workload with a processed ML API trace gets ml-api-trace.csv."""
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        torch_operator=None,
        triton_operator=None,
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(args)
    inst._report_runs = {"/w1": schema.Workload(), "/w2": schema.Workload()}
    trace = pd.DataFrame({"Function": ["aten::addmm"]})
    inst._runs["/w2"].ml_api_trace_pairs = [object()]
    inst._runs["/w2"].ml_api_trace_df = trace
    saved = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.write_ml_api_trace_csv",
        lambda folder, frame: saved.append((folder, frame)),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda *a, **kw: None,
    )

    make_workload_folders(tmp_path, ["/w1", "/w2"])

    inst.write_outputs(tmp_path)

    assert [folder.name for folder, _frame in saved] == ["2-w2"]
    assert saved[0][1] is trace
