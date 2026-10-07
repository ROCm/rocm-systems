# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_cli import cli_analysis

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


@pytest.mark.torch_ops
def test_warn_ml_api_trace_errors_lists_all(monkeypatch):
    """Accumulated ML API errors are printed after the call tree."""
    from rocprof_compute_analyze.analysis_cli import _warn_ml_api_trace_errors
    from utils.ml_api_trace_errors import (
        MissingSourceLocationError,
        UncorrelatedLauncherIntervalError,
    )
    from utils.schema import Workload

    seen = []
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_cli.console_warning",
        lambda *argv: seen.append(argv),
    )
    workload = Workload()
    workload.ml_api_trace_errors = [
        MissingSourceLocationError("aten::detach", "1", 0.0),
        UncorrelatedLauncherIntervalError("eval", "2", 1.0, 2.0, "9"),
    ]
    _warn_ml_api_trace_errors(workload)
    assert seen[0] == ("analysis", "2 ML API trace error(s):")
    assert "aten::detach" in seen[1][1]
    assert "Uncorrelated launcher interval" in seen[2][1]
