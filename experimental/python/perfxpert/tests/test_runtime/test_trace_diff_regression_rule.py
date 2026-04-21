###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for the ``trace_diff_regression_rule`` gate rule (Confluence row #7).

The rule wraps the shared ``tools.trace_diff.diff_runs`` engine and
applies the configurable ``PERFXPERT_CI_REGRESSION_THRESHOLD`` gate
threshold (default 5%).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from perfxpert.runtime.gate_cascade import (
    CI_REGRESSION_ENV,
    DEFAULT_CI_REGRESSION_THRESHOLD_PCT,
    _resolve_ci_threshold,
    trace_diff_regression_rule,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_DB = FIXTURES / "regression_baseline.db"
REGRESSED_DB = FIXTURES / "regression_tail_hurt.db"
IMPROVED_DB = FIXTURES / "regression_improved.db"


@pytest.fixture(scope="module")
def _fixtures_exist():
    missing = [p for p in (BASELINE_DB, REGRESSED_DB, IMPROVED_DB) if not p.exists()]
    if missing:
        pytest.skip(f"fixtures missing: {missing}")
    return True


def test_resolve_ci_threshold_default():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(CI_REGRESSION_ENV, None)
        assert _resolve_ci_threshold(None) == DEFAULT_CI_REGRESSION_THRESHOLD_PCT


def test_resolve_ci_threshold_env_override():
    with mock.patch.dict(os.environ, {CI_REGRESSION_ENV: "10.5"}):
        assert _resolve_ci_threshold(None) == 10.5


def test_resolve_ci_threshold_cli_wins_over_env():
    with mock.patch.dict(os.environ, {CI_REGRESSION_ENV: "10.0"}):
        assert _resolve_ci_threshold(0.5) == 0.5


def test_resolve_ci_threshold_invalid_env_falls_back():
    with mock.patch.dict(os.environ, {CI_REGRESSION_ENV: "not-a-number"}):
        assert _resolve_ci_threshold(None) == DEFAULT_CI_REGRESSION_THRESHOLD_PCT


def test_gate_flags_regression_above_threshold(_fixtures_exist):
    """Tail-hurt fixture regresses ~4.5% — above a 2% explicit threshold."""
    v = trace_diff_regression_rule(
        str(BASELINE_DB), str(REGRESSED_DB), threshold_pct=2.0
    )
    assert v["rule"] == "trace_diff_regression"
    assert v["verdict"] == "fail"
    assert v["wall_delta_pct"] > 2.0
    assert "regressed" in v["reason"].lower()
    assert v["diff"]["schema_version"] == "0.3.1"


def test_gate_passes_improvement(_fixtures_exist):
    """Improved fixture — gate must pass."""
    v = trace_diff_regression_rule(str(BASELINE_DB), str(IMPROVED_DB))
    assert v["verdict"] == "pass"
    assert v["wall_delta_pct"] < 0.0


def test_gate_env_threshold_override_relaxes_verdict(_fixtures_exist):
    """With a generous threshold the tail-hurt regression no longer trips
    the gate."""
    with mock.patch.dict(os.environ, {CI_REGRESSION_ENV: "50.0"}):
        v = trace_diff_regression_rule(str(BASELINE_DB), str(REGRESSED_DB))
        assert v["verdict"] == "pass"


def test_gate_explicit_threshold_takes_precedence(_fixtures_exist):
    """Explicit ``threshold_pct`` beats ``$PERFXPERT_CI_REGRESSION_THRESHOLD``."""
    with mock.patch.dict(os.environ, {CI_REGRESSION_ENV: "99.0"}):
        v = trace_diff_regression_rule(
            str(BASELINE_DB), str(REGRESSED_DB), threshold_pct=0.1
        )
        assert v["verdict"] == "fail"
