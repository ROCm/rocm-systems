# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import pytest


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
