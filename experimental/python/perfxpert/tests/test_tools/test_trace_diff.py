###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for the ``trace_diff`` READ_ONLY tool (Confluence row #7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from perfxpert.tools._class import ToolClass
from perfxpert.tools.trace_diff import diff_runs


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_DB = FIXTURES / "regression_baseline.db"
# NOTE: use ``regression_tail_hurt.db`` as the "regressed" counterpart —
# it uses the same legacy schema as the baseline. ``regressed.db`` is a
# full rocprofv3 multi-table database built for a different test target
# (kernel runtimes sourced from the info-kernel join), not a direct
# kernel-for-kernel pair with ``regression_baseline.db``.
REGRESSED_DB = FIXTURES / "regression_tail_hurt.db"
IMPROVED_DB = FIXTURES / "regression_improved.db"


@pytest.fixture(scope="module")
def _fixtures_exist():
    missing = [p for p in (BASELINE_DB, REGRESSED_DB, IMPROVED_DB) if not p.exists()]
    if missing:
        pytest.skip(f"fixtures missing: {missing}")
    return True


def test_diff_runs_is_read_only():
    assert getattr(diff_runs, "__tool_class__", None) is ToolClass.READ_ONLY


def test_trace_diff_tool_returns_expected_shape(_fixtures_exist):
    result = diff_runs(str(BASELINE_DB), str(REGRESSED_DB))
    # Top-level shape contract.
    assert result["schema_version"] == "0.3.1"
    assert result["baseline_db"] == str(BASELINE_DB)
    assert result["new_db"] == str(REGRESSED_DB)
    assert isinstance(result["wall_delta_ns"], int)
    assert isinstance(result["wall_delta_pct"], float)
    assert isinstance(result["per_kernel"], list)
    assert isinstance(result["primary_regressions"], list)
    assert isinstance(result["primary_improvements"], list)
    assert isinstance(result["narrative"], str) and result["narrative"]

    # Every per_kernel entry carries the full contract.
    for k in result["per_kernel"]:
        assert set(k.keys()) >= {
            "name",
            "baseline_ns",
            "new_ns",
            "delta_ns",
            "delta_pct",
            "regressed",
            "was_hot",
        }
        assert isinstance(k["name"], str)
        assert isinstance(k["baseline_ns"], int)
        assert isinstance(k["new_ns"], int)
        assert isinstance(k["delta_pct"], float)
        assert isinstance(k["regressed"], bool)
        assert isinstance(k["was_hot"], bool)


def test_trace_diff_detects_regression(_fixtures_exist):
    """The regressed fixture should surface wall-time regression."""
    result = diff_runs(str(BASELINE_DB), str(REGRESSED_DB))
    assert result["wall_delta_pct"] > 3.0, (
        f"expected wall regression > 3%; got {result['wall_delta_pct']}%"
    )
    # At least one primary regression must surface.
    assert result["primary_regressions"], "no regressions detected in regressed fixture"


def test_trace_diff_detects_improvement(_fixtures_exist):
    """Baseline vs improved.db — matmul shrunk 20% ⇒ negative wall pct."""
    result = diff_runs(str(BASELINE_DB), str(IMPROVED_DB))
    # Wall time should have shrunk (negative pct).
    assert result["wall_delta_pct"] < 0.0, (
        f"expected wall improvement; got {result['wall_delta_pct']}%"
    )
    assert result["primary_improvements"], (
        "no improvements detected in regression_improved fixture"
    )


def test_trace_diff_narrative_is_deterministic(_fixtures_exist):
    r1 = diff_runs(str(BASELINE_DB), str(REGRESSED_DB))
    r2 = diff_runs(str(BASELINE_DB), str(REGRESSED_DB))
    assert r1["narrative"] == r2["narrative"]
    assert r1["wall_delta_ns"] == r2["wall_delta_ns"]


def test_trace_diff_top_kernels_limit(_fixtures_exist):
    result = diff_runs(str(BASELINE_DB), str(REGRESSED_DB), top_kernels=2)
    assert len(result["per_kernel"]) <= 2
