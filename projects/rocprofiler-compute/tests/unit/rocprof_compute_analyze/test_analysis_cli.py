# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
import logging
from functools import partial
from types import SimpleNamespace
from typing import List, Tuple

import pandas as pd
import pytest

from ml_api_tracing.analyze.definitions import CallTreeNode, KernelStats
from ml_api_tracing.analyze.orchestrator import apply_operator_filter
from ml_api_tracing.analyze.output import list_operators, show_matched_operator_report
from ml_api_tracing.marker_format import compose_marker
from rocprof_compute_analyze.analysis_cli import cli_analysis
from utils import parser, schema

RMSNORM_PMC_GROUP_KEY = (4, "rmsnorm_kernel", 0, 0, 0)
ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
OPERATOR_TRACE_GUID = "00000000-0000-0000-0000-000000000001"


def triton_rmsnorm_forest():
    rmsnorm = CallTreeNode(name="triton.JITFunction.rmsnorm_kernel", backend="triton")
    rmsnorm.kernels["rmsnorm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=20.0,
        pmc_group_keys=[RMSNORM_PMC_GROUP_KEY],
    )
    return {"1": [rmsnorm]}


def build_operator_preprocessing_frames(child_backend):
    marker_rows = [
        {
            "Function": compose_marker(
                operator_name, "model.py:1", backend=backend, args="()"
            ),
            "Thread_Id": 7,
            "Correlation_ID": correlation_id,
            "GUID": OPERATOR_TRACE_GUID,
            "Start_Timestamp": start,
            "End_Timestamp": end,
        }
        for operator_name, backend, correlation_id, start, end in [
            ("nn.Module.SimpleModel.forward", "torch", 1, 0, 1000),
            ("aten::addmm", child_backend, 2, 100, 500),
            ("aten::relu", child_backend, 3, 600, 900),
        ]
    ]
    counter_rows = [
        {
            "Correlation_ID": correlation_id,
            "GUID": OPERATOR_TRACE_GUID,
            "Dispatch_ID": dispatch_id,
            "Kernel_Name": kernel_name,
            "Grid_Size": 256,
            "Workgroup_Size": 64,
            "LDS_Per_Workgroup": 0,
            "Start_Timestamp": start,
            "End_Timestamp": end,
        }
        for correlation_id, dispatch_id, kernel_name, start, end in [
            (2, 101, "addmm_kernel", 150, 200),
            (2, 102, "addmm_kernel", 250, 350),
            (3, 201, "relu_kernel", 650, 675),
        ]
    ]
    return pd.DataFrame(marker_rows), pd.DataFrame(counter_rows)


def write_operator_preprocessing_trace(workload_path, child_backend="torch"):
    marker_frame, counter_frame = build_operator_preprocessing_frames(child_backend)
    marker_frame.to_csv(
        workload_path / "ml_api_trace_pmc_perf_0_marker_api_trace.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8",
    )
    counter_frame.to_csv(
        workload_path / "ml_api_trace_pmc_perf_0_counter_collection.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8",
    )


def load_operator_pmc_with_slots(boundary, *args, **kwargs):
    workload = boundary.workload
    boundary.stages["pmc"] = {
        "forest": workload.ml_api_call_trees,
        "dispatch_ids": workload.filter_dispatch_ids,
    }
    return boundary.raw_pmc, boundary.slot_to_id, {}


def rank_operator_kernels(boundary, **kwargs):
    boundary.stages["ranking"] = {
        "raw_pmc": kwargs["df_in"],
        "dispatch_ids": list(kwargs["filter_dispatch_ids"]),
        "dispatch_filter": boundary.workload.filter_dispatch_ids,
    }
    return boundary.kernel_top, boundary.dispatch_table


def load_operator_tables(boundary, **kwargs):
    workload = boundary.workload
    boundary.stages["tables"] = {
        "workload": kwargs["workload"],
        "dispatch_ids": list(workload.filter_dispatch_ids),
        "kernel_ids": dict(workload.operator_kernel_ids),
    }


@pytest.fixture
def operator_preprocessing(tmp_path, monkeypatch):
    """Run real ML adapters with only generic analyze boundaries isolated."""
    workload_path = tmp_path / "workload"
    workload_path.mkdir()
    write_operator_preprocessing_trace(workload_path)
    args = argparse.Namespace(
        path=[[str(workload_path)]],
        verbose=0,
        time_unit="ns",
        operator=["torch:*SimpleModel*"],
        list_operators=None,
        gpu_kernel=[],
    )
    workload = schema.Workload(sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]))
    cli = cli_analysis.__new__(cli_analysis)
    cli._profiling_config = {"trace_ml_api": ["torch", "triton"]}
    cli._runs = {str(workload_path): workload}
    cli._arch_configs = {"gfx950": SimpleNamespace(dfs_expressions={})}
    cli._OmniAnalyze_Base__args = args
    boundary = SimpleNamespace(
        cli=cli,
        args=args,
        workload=workload,
        path=workload_path,
        raw_pmc=pd.DataFrame({
            "Dispatch_ID": [8, 10, 3, 99],
            "Kernel_Name": [
                "addmm_kernel",
                "addmm_kernel",
                "relu_kernel",
                "outside_kernel",
            ],
        }),
        slot_to_id={
            (101, "addmm_kernel", 256, 64, 0): 8,
            (102, "addmm_kernel", 256, 64, 0): 10,
            (201, "relu_kernel", 256, 64, 0): 3,
        },
        kernel_top=pd.DataFrame({"Kernel_Name": ["relu_kernel", "addmm_kernel"]}),
        dispatch_table=pd.DataFrame({"Dispatch_ID": [3, 8, 10]}),
        stages={},
    )

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.pre_processing",
        lambda self: None,
    )
    monkeypatch.setattr(cli, "pc_sampling_only", lambda: False)
    monkeypatch.setattr(cli, "load_pc_sampling_tool_data", lambda path: None)
    monkeypatch.setattr(
        "utils.file_io.create_df_pmc_with_slot_ids",
        partial(load_operator_pmc_with_slots, boundary),
    )
    monkeypatch.setattr(
        "utils.file_io.create_df_kernel_top_stats",
        partial(rank_operator_kernels, boundary),
    )
    monkeypatch.setattr(
        "utils.parser.load_table_data", partial(load_operator_tables, boundary)
    )
    return boundary


def error_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]


def workload_with_operator_forest(simple_model_forest_with_relu_and_addmm):
    workload = schema.Workload()
    workload.ml_api_call_trees = simple_model_forest_with_relu_and_addmm()
    return workload


def cli_with_two_operator_workloads(args, simple_model_forest_with_relu_and_addmm):
    inst = cli_analysis.__new__(cli_analysis)
    inst._profiling_config = {"trace_ml_api": ["torch"]}
    inst._output = None
    inst._runs = {
        "/w1": workload_with_operator_forest(simple_model_forest_with_relu_and_addmm),
        "/w2": workload_with_operator_forest(simple_model_forest_with_relu_and_addmm),
    }
    for workload in inst._runs.values():
        workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx950"}])
    inst._arch_configs = {"gfx950": SimpleNamespace(dfs_expressions={})}
    inst._OmniAnalyze_Base__args = args
    return inst


def write_selector_lifecycle_workload(parent, name, backend):
    workload = parent / name / "GPU"
    workload.mkdir(parents=True)
    (workload / "profiling_config.yaml").write_text(
        f"trace_ml_api:\n- {backend}\n", encoding="utf-8"
    )
    pd.DataFrame({"SQ_WAVES": [1]}).to_csv(
        workload / "results_0.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame([{"gpu_arch": "gfx950"}]).to_csv(workload / "sysinfo.csv", index=False)
    return workload


def generate_selector_lifecycle_configs(analyzer, arch, *_args, **_kwargs):
    """Isolate metric definitions while retaining real base workload initialization."""
    analyzer._arch_configs[arch] = SimpleNamespace(dfs={}, dfs_type={})
    return analyzer._arch_configs


def selector_lifecycle_args(paths, listing, operator):
    return argparse.Namespace(
        path=[[str(path)] for path in paths],
        list_operators=listing,
        operator=operator,
        output_name=None,
        output_format="stdout",
        config_dir="",
        list_stats=False,
        filter_metrics=None,
        normal_unit="per_kernel",
        gpu_kernel=None,
        gpu_id=None,
        gpu_dispatch_id=None,
        specs_correction=None,
        no_roof=True,
        verbose=0,
        time_unit="ns",
    )


def build_selector_lifecycle_pc_workload(workload, _path, _args, tool_data):
    """Supply PC scaffolding while the real lifecycle decides whether to use it."""
    assert tool_data == []
    workload.raw_pmc = pd.DataFrame({"Dispatch_ID": [7]})
    workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = pd.DataFrame({
        "Kernel_Name": ["pc_kernel"]
    })


@pytest.mark.parametrize(
    ("listing", "operator"),
    [
        pytest.param([], ["invalid-prefix:*"], id="bare-listing-precedence"),
        pytest.param(["torch"], None, id="untraced-listing"),
        pytest.param(None, ["invalid-prefix:*"], id="invalid-filter"),
        pytest.param(None, ["torch:*"], id="untraced-filter"),
    ],
)
def test_pc_only_workloads_preserve_operator_lifecycle(
    tmp_path, monkeypatch, capsys, listing, operator
):
    paths = [
        write_selector_lifecycle_workload(tmp_path, "first", "torch"),
        write_selector_lifecycle_workload(tmp_path, "second", "triton"),
    ]
    (paths[0] / "profiling_config.yaml").write_text(
        "filter_blocks:\n- pc_sampling\n", encoding="utf-8"
    )
    args = selector_lifecycle_args(paths, listing, operator)
    analyzer = cli_analysis(args, {})
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.generate_configs",
        generate_selector_lifecycle_configs,
    )
    monkeypatch.setattr(
        analyzer,
        "build_pc_sampling_only_workload",
        build_selector_lifecycle_pc_workload,
    )
    displayed_workloads = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda _args, workloads, *_rest: displayed_workloads.extend(workloads),
    )

    analyzer.sanitize()
    if listing is not None:
        with pytest.raises(SystemExit) as error:
            analyzer.pre_processing()
        assert error.value.code == 0
    else:
        analyzer.pre_processing()
        analyzer.run_analysis()
        assert displayed_workloads == [str(path) for path in paths]

    assert analyzer.pc_sampling_only()
    assert set(analyzer._runs) == {str(path) for path in paths}
    for workload in analyzer._runs.values():
        assert workload.raw_pmc["Dispatch_ID"].tolist() == [7]
        assert workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID]["Kernel_Name"].tolist() == [
            "pc_kernel"
        ]
        assert workload.operator_patterns == {}
        assert workload.ml_api_call_trees == {}
    assert args.list_operators == listing
    assert args.operator == operator
    assert "Matched" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("backend_mode", "dispatch_filter", "kernel_filter", "ranked_ids", "final_ids"),
    [
        pytest.param("torch", [], [], [3, 8, 10], [3, 8, 10], id="unrestricted"),
        pytest.param(
            "triton",
            ["99", "10", "3", "8", "8"],
            [1],
            [3, 8, 10],
            [8, 10],
            id="explicit-dispatch-and-sliced-kernel",
        ),
        pytest.param(
            "combined",
            [" > 8 "],
            [],
            [10],
            [10],
            id="combined-backends-exclusive-threshold",
        ),
    ],
)
def test_pre_processing_operator_filters_publish_intersections_and_numbering(
    operator_preprocessing,
    caplog,
    backend_mode,
    dispatch_filter,
    kernel_filter,
    ranked_ids,
    final_ids,
):
    boundary = operator_preprocessing
    workload = boundary.workload
    workload.filter_dispatch_ids = dispatch_filter
    workload.filter_kernel_ids = kernel_filter
    boundary.args.gpu_kernel = [[str(kernel_id) for kernel_id in kernel_filter]]
    if backend_mode != "torch":
        write_operator_preprocessing_trace(boundary.path, child_backend="triton")
    if backend_mode == "combined":
        boundary.args.operator = ["torch:*SimpleModel*", "triton:aten::*"]
    if backend_mode == "triton":
        boundary.args.operator = ["triton:aten::*"]
    caplog.set_level(logging.INFO)

    boundary.cli.pre_processing()

    assert workload.ml_api_trace_df["Operator_Name"].tolist() == [
        "nn.Module.SimpleModel.forward",
        "aten::addmm",
        "aten::relu",
    ]
    assert workload.operator_kernel_ids == {"relu_kernel": 0, "addmm_kernel": 1}
    assert workload.filter_dispatch_ids == final_ids
    assert workload.raw_pmc is boundary.raw_pmc
    assert boundary.stages["pmc"]["dispatch_ids"] is dispatch_filter
    assert boundary.stages["pmc"]["forest"] is workload.ml_api_call_trees
    assert boundary.stages["ranking"]["raw_pmc"] is boundary.raw_pmc
    assert boundary.stages["ranking"]["dispatch_ids"] == ranked_ids
    assert boundary.stages["tables"] == {
        "workload": workload,
        "dispatch_ids": final_ids,
        "kernel_ids": {"relu_kernel": 0, "addmm_kernel": 1},
    }
    assert workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] is boundary.kernel_top
    assert workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] is boundary.dispatch_table
    assert any(
        f"selected {len(ranked_ids)} dispatch(es) for metric analysis."
        in record.getMessage()
        for record in caplog.records
    )
    assert error_messages(caplog) == []


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
        operator=None,
        list_operators=None,
    )
    inst._OmniAnalyze_Base__args = args

    membw_calls: List[Tuple] = []
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
    monkeypatch.setattr(
        "utils.file_io.create_df_pmc", lambda *a, **kw: (pd.DataFrame(), {})
    )
    monkeypatch.setattr(
        "utils.file_io.create_df_kernel_top_stats",
        lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr("utils.parser.load_table_data", lambda *a, **kw: None)

    inst.pre_processing()

    assert len(membw_calls) == (1 if expect_called else 0)


def test_list_operators_joint_backend_heading(capsys, torch_parent_triton_child_forest):
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    kernel_top = pd.DataFrame({
        "Kernel_Name": ["torch_gemm_kernel", "triton_matmul_kernel"]
    })
    list_operators("/workload", workload, kernel_top, ["torch", "triton"])
    captured = capsys.readouterr()
    assert "PyTorch, Triton Operator Call Tree" in captured.out


def test_show_operator_report_prints_matched_subtree(
    capsys, simple_model_forest_with_relu_and_addmm
):
    workload = workload_with_operator_forest(simple_model_forest_with_relu_and_addmm)
    workload.operator_patterns = {"torch": ["*addmm*"]}
    apply_operator_filter(workload, "/workload")
    show_matched_operator_report(workload)
    captured = capsys.readouterr()
    assert "Matched PyTorch Operators: *addmm*" in captured.out
    assert "aten::addmm" in captured.out


def test_list_operators_omits_other_backend_kernel_lines(
    capsys, torch_parent_triton_child_forest
):
    workload = schema.Workload()
    workload.ml_api_call_trees = torch_parent_triton_child_forest()
    kernel_top = pd.DataFrame({
        "Kernel_Name": ["torch_gemm_kernel", "triton_matmul_kernel"]
    })
    list_operators("/workload", workload, kernel_top, ["triton"])
    captured = capsys.readouterr()
    assert "triton_matmul_kernel" in captured.out
    assert "torch_gemm_kernel" not in captured.out


@pytest.mark.parametrize("list_selectors", [[], ["torch"]], ids=["bare", "named"])
def test_pre_processing_lists_operators_for_every_path(
    monkeypatch, simple_model_forest_with_relu_and_addmm, list_selectors
):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        verbose=0,
        time_unit="ns",
        operator=["invalid-prefix:ignored"],
        list_operators=list_selectors,
    )
    inst = cli_with_two_operator_workloads(
        args, simple_model_forest_with_relu_and_addmm
    )
    listed_paths = []
    monkeypatch.setattr(
        "utils.file_io.load_profiling_config",
        lambda _path: {"trace_ml_api": ["torch"]},
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
    monkeypatch.setattr(
        "utils.file_io.create_df_pmc", lambda *a, **kw: (pd.DataFrame(), {})
    )
    monkeypatch.setattr(
        "utils.file_io.create_df_kernel_top_stats",
        lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.process_ml_api_trace_output",
        lambda workload, workload_dir: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.list_operators",
        lambda workload_path, workload, kernel_top, backends: listed_paths.append(
            workload_path
        ),
    )
    with pytest.raises(SystemExit) as excinfo:
        inst.pre_processing()
    assert excinfo.value.code == 0
    assert listed_paths == ["/w1", "/w2"]
    assert args.operator == ["invalid-prefix:ignored"]
    assert args.list_operators == list_selectors


def test_run_analysis_prints_matched_tree_for_every_path(
    monkeypatch, simple_model_forest_with_relu_and_addmm
):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        operator=["*addmm*"],
        list_operators=None,
        list_stats=False,
    )
    inst = cli_with_two_operator_workloads(
        args, simple_model_forest_with_relu_and_addmm
    )
    reported_workloads = []

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.run_analysis",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.show_matched_operator_report",
        reported_workloads.append,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all",
        lambda *a, **kw: None,
    )
    inst.run_analysis()
    assert reported_workloads == [inst._runs["/w1"], inst._runs["/w2"]]


def test_apply_operator_filter_unscoped_mixed_paths(
    simple_model_forest_with_relu_and_addmm,
):
    """Bare --operator on sn_torch then tf_triton binds each path's yaml."""
    torch_workload = workload_with_operator_forest(
        simple_model_forest_with_relu_and_addmm
    )
    torch_workload.operator_patterns = {"torch": ["**"], "user": ["**"]}
    triton_workload = schema.Workload()
    triton_workload.ml_api_call_trees = triton_rmsnorm_forest()
    triton_workload.operator_patterns = {"triton": ["**"], "user": ["**"]}
    torch_keys = apply_operator_filter(torch_workload, "/sn_torch")
    triton_keys = apply_operator_filter(triton_workload, "/tf_triton")
    assert ADDMM_PMC_GROUP_KEY in torch_keys
    assert triton_keys == [RMSNORM_PMC_GROUP_KEY]


def test_run_analysis_reports_each_workload_bound_map(
    capsys, monkeypatch, simple_model_forest_with_relu_and_addmm
):
    args = argparse.Namespace(
        path=[["/w1"], ["/w2"]],
        operator=[],
        list_operators=None,
        list_stats=False,
    )
    cli = cli_with_two_operator_workloads(args, simple_model_forest_with_relu_and_addmm)
    cli._runs["/w2"].ml_api_call_trees = triton_rmsnorm_forest()
    cli._runs["/w1"].operator_patterns = {"torch": ["**"], "user": ["**"]}
    cli._runs["/w2"].operator_patterns = {"triton": ["**"], "user": ["**"]}
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_base.OmniAnalyze_Base.run_analysis",
        lambda self: None,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.tty.show_all", lambda *args: None
    )

    cli.run_analysis()

    report = capsys.readouterr().out
    assert [line for line in report.splitlines() if line.startswith("Matched ")] == [
        "Matched PyTorch, User Operators: **",
        "Matched Triton, User Operators: **",
    ]
    assert "triton.JITFunction.rmsnorm_kernel" in report
    assert "aten::addmm" in report
