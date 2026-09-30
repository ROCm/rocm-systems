# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import cli_analysis, parse_operator_patterns
from utils import parser, schema
from utils.analysis_orm import Database
from utils.metrics import expression_evaluator
from utils.roofline_calc import RooflineBenchmark


def make_roofline_analyzer(tmp_path):
    """Build a single-path CLI analyzer with selected and unselected dispatches."""
    path = "/workloads/vector/run"
    args = argparse.Namespace(
        path=[[path]],
        list_stats=False,
        sort="kernels",
        mem_level=["ALL"],
        roofline_data_type=["FP32"],
        gpu_kernel=[[0]],
        debug=False,
        time_unit="ns",
    )
    analyzer = cli_analysis(args, {})
    analyzer._output_dir = tmp_path
    analyzer._profiling_config = {}
    workload = schema.Workload(
        sys_info=pd.DataFrame([
            {
                "gpu_arch": "gfx90a",
                "gpu_series": "MI200",
                "gpu_model": "MI200",
            }
        ])
    )
    workload.raw_pmc = pd.DataFrame({
        "Kernel_Name": ["selected", "other", "different_gpu"],
        "GPU_ID": [0, 0, 1],
        "Dispatch_ID": [1, 2, 3],
        "Start_Timestamp": [0, 0, 0],
        "End_Timestamp": [10, 20, 30],
    })
    workload.filter_gpu_ids = ["0"]
    workload.filter_kernel_ids = [0]
    workload.filter_dispatch_ids = None
    analyzer._runs = {path: workload}
    analyzer._arch_configs = {"gfx90a": schema.ArchConfig()}
    analyzer.set_soc({
        "gfx90a": SimpleNamespace(_mspec=object(), get_args=lambda: args)
    })
    return analyzer, workload


def patch_shared_roofline(monkeypatch):
    """Record the shared pipeline while returning distinguishable queried values."""
    module = "rocprof_compute_analyze.analysis_cli"
    benchmark = RooflineBenchmark(0, {"FP32Flops": 100, "HBMBw": 200}, "MFMA")
    result = SimpleNamespace(
        benchmark=benchmark, kernels=[SimpleNamespace(kernel_name="selected")]
    )
    view = SimpleNamespace(
        benchmark_peaks={"FP32Flops": 999},
        plot_points={"queried": True},
        tty_tables=MagicMock(return_value={0: {"queried_table": True}}),
    )
    compute = MagicMock(return_value=result)
    persist = MagicMock()
    load = MagicMock(return_value=view)
    events = []
    session = SimpleNamespace(add=lambda row: events.append("add"))
    monkeypatch.setattr(f"{module}.validate_roofline_csv", lambda path: (True, ""))
    monkeypatch.setattr(f"{module}.load_roofline_benchmark", lambda *args: benchmark)
    monkeypatch.setattr(f"{module}.compute_roofline", compute, raising=False)
    monkeypatch.setattr(f"{module}.persist_roofline", persist, raising=False)
    monkeypatch.setattr(f"{module}.load_roofline_view", load, raising=False)
    monkeypatch.setattr(Database, "init", lambda: events.append("init"))
    monkeypatch.setattr(Database, "get_session", lambda: session)
    monkeypatch.setattr(Database, "commit", lambda: events.append("commit"))
    monkeypatch.setattr(Database, "close", lambda: events.append("close"))
    monkeypatch.setattr(
        "utils.mi_gpu_spec.mi_gpu_specs.get_memory_levels", lambda model: ["HBM"]
    )
    monkeypatch.setattr(
        "utils.parser.apply_filters",
        lambda workload, *args, **kwargs: workload.raw_pmc.iloc[:1],
    )
    return compute, persist, load, view, events


# -- parse_operator_patterns (torch_operator) -------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    args = argparse.Namespace(torch_operator=["relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu"]

    args = argparse.Namespace(torch_operator=["relu", "conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    args = argparse.Namespace(torch_operator=["relu,conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    args = argparse.Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    result = parse_operator_patterns(args, "torch_operator")
    assert result == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    parse = parse_operator_patterns
    assert parse(argparse.Namespace(torch_operator=[]), "torch_operator") == ["**"]
    assert parse(argparse.Namespace(torch_operator=None), "torch_operator") == []
    assert parse(argparse.Namespace(), "torch_operator") == []


# -- parse_operator_patterns / triton backend selection ---------------------


@pytest.mark.torch_ops
def test_parse_operator_patterns_generic_attr():
    """parse_operator_patterns reads the given dest attribute."""
    args = argparse.Namespace(
        triton_operator=["*matmul*,*softmax*"], torch_operator=None
    )
    assert parse_operator_patterns(args, "triton_operator") == [
        "*matmul*",
        "*softmax*",
    ]
    assert parse_operator_patterns(args, "triton_operator") != parse_operator_patterns(
        args, "torch_operator"
    )
    assert parse_operator_patterns(
        argparse.Namespace(triton_operator=[]), "triton_operator"
    ) == ["**"]


@pytest.mark.torch_ops
def test_filter_by_backend_selects_only_requested_backend():
    df = pd.DataFrame({
        "Operator_Name": ["aten::mm", "triton_matmul", "aten::relu"],
        "Backend": ["torch", "triton", "torch"],
    })

    triton_df = cli_analysis._filter_by_backend(df, "triton")
    assert triton_df["Operator_Name"].tolist() == ["triton_matmul"]

    torch_df = cli_analysis._filter_by_backend(df, "torch")
    assert torch_df["Operator_Name"].tolist() == ["aten::mm", "aten::relu"]


@pytest.mark.torch_ops
def test_filter_by_backend_without_column_defaults_to_torch():
    df = pd.DataFrame({"Operator_Name": ["aten::mm", "aten::relu"]})

    # Without a Backend column, rows are treated as torch.
    assert len(cli_analysis._filter_by_backend(df, "torch")) == 2
    assert cli_analysis._filter_by_backend(df, "triton").empty


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    args = argparse.Namespace(torch_operator=["*"])
    assert parse_operator_patterns(args, "torch_operator") == ["*"]

    args = argparse.Namespace(torch_operator=["*,torch.relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["*", "torch.relu"]


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
        random_port=False,
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


def test_operator_filter_regenerates_trace_in_analysis_directory(tmp_path, monkeypatch):
    """Operator selection uses raw traces even when a workload has cached output."""
    workload_path = tmp_path / "workload"
    cache_dir = workload_path / "ml_api_trace"
    cache_dir.mkdir(parents=True)
    cached_trace = cache_dir / "consolidated.csv"
    cached_trace.write_text("stale workload cache", encoding="utf-8")
    analyzer = cli_analysis.__new__(cli_analysis)
    analyzer._output_dir = tmp_path / "analysis"
    trace = pd.DataFrame({
        "Operator_Name": ["torch.relu"],
        "Kernel_Name": ["relu_kernel"],
        "Backend": ["torch"],
    })
    calls = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.process_ml_api_trace_output",
        lambda source, output: (
            calls.append((source, output)) or trace,
            output / "ml_api_trace",
        ),
    )
    writes = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.write_ml_api_trace_consolidated_csv",
        lambda frame, path: writes.append(path),
    )
    workload = SimpleNamespace(
        dfs={
            parser.PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({
                "Kernel_Name": ["other_kernel", "relu_kernel"],
            })
        },
        filter_kernel_ids=[],
        matched_ml_api_trace_dfs={},
    )
    analyzer.apply_operator_filter(
        argparse.Namespace(torch_operator=["torch.relu"]),
        workload,
        str(workload_path),
        "torch",
    )
    assert calls == [(str(workload_path), analyzer._output_dir)]
    assert writes == [analyzer._output_dir / "ml_api_trace"]
    assert workload.filter_kernel_ids == [1]
    assert cached_trace.read_text(encoding="utf-8") == "stale workload cache"


def test_cli_roofline_renders_only_queried_view_data(tmp_path, monkeypatch):
    """Terminal plots, HTML, and TTY tables share queried database values."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    renderer = MagicMock()
    renderer.get_dtype.return_value = ["FP32"]
    renderer.construct_plotly_figures.return_value = (None, None, "", "")
    renderer_factory = MagicMock(return_value=renderer)
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.Roofline", renderer_factory
    )
    show_all = MagicMock()
    monkeypatch.setattr("utils.tty.show_all", show_all)
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.calc_ai_analyze",
        lambda **kwargs: pytest.fail("legacy roofline computation called"),
        raising=False,
    )
    analyzer.run_analysis()
    assert compute.call_count == 1
    assert compute.call_args.kwargs["evaluate"] is expression_evaluator.evaluate
    assert compute.call_args.kwargs["pmc_df"]["Kernel_Name"].tolist() == ["selected"]
    assert compute.call_args.kwargs["stats_df"]["Kernel_Name"].tolist() == [
        "selected",
        "other",
    ]
    assert persist.call_count == 1
    assert set(persist.call_args.args[2]) == {"selected"}
    assert load.call_count == 1
    assert renderer_factory.call_args.kwargs["benchmark_peaks"] is view.benchmark_peaks
    renderer.cli_generate_plot.assert_called_once_with(
        dtype="FP32", ai_data=view.plot_points
    )
    renderer.construct_plotly_figures.assert_called_once_with(ai_data=view.plot_points)
    assert renderer.save_html_files.call_args.kwargs["output_dir"] == tmp_path
    view.tty_tables.assert_called_once_with(analyzer._arch_configs["gfx90a"], [0])
    assert workload.roofline_metrics == {0: {"queried_table": True}}
    assert events[0] == "init"
    assert events[-2:] == ["commit", "close"]
    assert (
        show_all.call_args.kwargs["roof_plot"]
        == renderer.cli_generate_plot.return_value
    )


def test_cli_roofline_closes_inmemory_database_on_render_failure(tmp_path, monkeypatch):
    """An HTML failure still releases the temporary database session."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    renderer = MagicMock()
    renderer.get_dtype.return_value = ["FP32"]
    renderer.construct_plotly_figures.side_effect = RuntimeError("render failed")
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.Roofline", lambda **kwargs: renderer
    )
    with pytest.raises(RuntimeError, match="render failed"):
        analyzer.run_analysis()
    assert events[-1] == "close"


@pytest.mark.parametrize(
    "skip_reason",
    ["multiple_paths", "unsupported", "invalid", "missing_device", "missing_soc"],
)
def test_cli_roofline_skips_without_opening_database(
    tmp_path, monkeypatch, skip_reason
):
    """Rejected roofline workloads still print their normal report."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    if skip_reason == "multiple_paths":
        analyzer.get_args().path.append(["/workloads/second/run"])
    elif skip_reason == "unsupported":
        workload.sys_info.loc[0, "gpu_arch"] = "gfx908"
        analyzer._arch_configs["gfx908"] = schema.ArchConfig()
    elif skip_reason == "invalid":
        monkeypatch.setattr(
            "rocprof_compute_analyze.analysis_cli.validate_roofline_csv",
            lambda path: (False, "missing file"),
        )
    elif skip_reason == "missing_device":
        monkeypatch.setattr(
            "rocprof_compute_analyze.analysis_cli.load_roofline_benchmark",
            lambda *args: None,
        )
    else:
        analyzer.set_soc({})
    show_all = MagicMock()
    monkeypatch.setattr("utils.tty.show_all", show_all)
    analyzer.run_analysis()
    assert events == []
    compute.assert_not_called()
    persist.assert_not_called()
    load.assert_not_called()
    assert show_all.call_args.kwargs["roof_plot"] is None


def test_cli_list_stats_skips_roofline_pipeline(tmp_path, monkeypatch):
    """Listing kernel statistics does not compute roofline or open its database."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    analyzer.get_args().list_stats = True
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    show_stats = MagicMock()
    monkeypatch.setattr("utils.tty.show_kernel_stats", show_stats)
    analyzer.run_analysis()
    assert events == []
    compute.assert_not_called()
    show_stats.assert_called_once()


@pytest.mark.parametrize("failing_stage", ["persist_roofline", "load_roofline_view"])
def test_cli_roofline_closes_database_on_pipeline_failure(
    tmp_path, monkeypatch, failing_stage
):
    """Insertion and query failures both release the in-memory database."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli." + failing_stage,
        MagicMock(side_effect=RuntimeError("pipeline failed")),
    )
    with pytest.raises(RuntimeError, match="pipeline failed"):
        analyzer.run_analysis()
    assert events[-1] == "close"


def test_cli_roofline_preserves_named_kernel_table_order(tmp_path, monkeypatch):
    """String kernel selections become queried ranks in the requested order."""
    analyzer, workload = make_roofline_analyzer(tmp_path)
    workload.filter_kernel_ids = [" other ", "selected"]
    compute, persist, load, view, events = patch_shared_roofline(monkeypatch)
    result = compute.return_value
    result.kernels = [
        SimpleNamespace(kernel_name="selected", kernel_rank=0),
        SimpleNamespace(kernel_name="other", kernel_rank=2),
    ]
    renderer = MagicMock()
    renderer.get_dtype.return_value = ["FP32"]
    renderer.construct_plotly_figures.return_value = (None, None, "", "")
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.Roofline", lambda **kwargs: renderer
    )
    monkeypatch.setattr("utils.tty.show_all", lambda *args, **kwargs: None)
    analyzer.run_analysis()
    view.tty_tables.assert_called_once_with(analyzer._arch_configs["gfx90a"], [2, 0])
