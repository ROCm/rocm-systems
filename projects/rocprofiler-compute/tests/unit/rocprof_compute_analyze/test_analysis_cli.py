# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import cli_analysis, parse_operator_patterns
from utils import parser

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
