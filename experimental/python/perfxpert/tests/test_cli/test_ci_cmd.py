###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for ``perfxpert ci`` (Confluence row #8)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def _run_ci(extra, env=None):
    """Invoke ``python -m perfxpert ci <extra>`` and return CompletedProcess."""
    cmd = [sys.executable, "-m", "perfxpert", "ci", *extra]
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=full_env,
    )


def test_ci_cmd_rc0_on_improvement(_fixtures_exist):
    """Improved DB → rc=0 (runtime shrinks)."""
    proc = _run_ci([str(BASELINE_DB), str(IMPROVED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] < 0


def test_ci_cmd_rc1_on_regression_above_threshold(_fixtures_exist):
    """Tail-hurt fixture regresses ~4.5% — above a 2% threshold → rc=1."""
    proc = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--threshold", "2.0",
         "--format", "json"],
    )
    assert proc.returncode == 1, (
        f"expected rc=1 on regression with --threshold 2.0; got rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] > 2.0
    assert "regressed" in proc.stderr.lower()


def test_ci_cmd_respects_env_threshold_override(_fixtures_exist):
    """With a permissive $PERFXPERT_CI_REGRESSION_THRESHOLD env var the
    ~4.5% tail-hurt regression no longer trips the gate.

    (Also, with no override the 5% default already covers +4.5% — but we
    force the CLI threshold to 2% and confirm the env var overrides it.)
    """
    # First: with a permissive env var, even a strict --threshold would be
    # clobbered if the env var takes precedence. To verify the semantics
    # we supply only the env var and confirm rc=0 (env=50% > 4.5%).
    proc = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "50.0"},
    )
    assert proc.returncode == 0, (
        f"env override should allow rc=0; got rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # And conversely: a stringent env threshold (0.1%) fails the same run.
    proc2 = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "0.1"},
    )
    assert proc2.returncode == 1


def test_ci_cmd_cli_threshold_wins_over_env(_fixtures_exist):
    """``--threshold`` on the CLI takes precedence over the env var."""
    proc = _run_ci(
        [
            str(BASELINE_DB),
            str(REGRESSED_DB),
            "--threshold",
            "0.1",
            "--format",
            "json",
        ],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "99.0"},
    )
    # Threshold = 0.1% — any regression trips it despite the 99% env var.
    assert proc.returncode == 1


def test_ci_cmd_text_format_shows_regression_summary(_fixtures_exist):
    """Default text format must surface a readable regression line."""
    proc = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--threshold", "2.0",
         "--format", "text"]
    )
    assert proc.returncode == 1
    assert "wall delta" in proc.stdout.lower() or "wall-time" in proc.stderr.lower()
    # Stderr should carry the one-line summary.
    assert "regressed" in proc.stderr.lower() or "runtime regressed" in proc.stderr.lower()
