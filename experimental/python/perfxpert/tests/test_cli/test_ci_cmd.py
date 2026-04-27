"""Tests for ``perfxpert ci``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BASELINE_DB = FIXTURES / "regression_baseline.db"
REGRESSED_DB = FIXTURES / "regression_tail_hurt.db"
IMPROVED_DB = FIXTURES / "regression_improved.db"


@pytest.fixture(scope="module")
def _fixtures_exist():
    missing = [path for path in (BASELINE_DB, REGRESSED_DB, IMPROVED_DB) if not path.exists()]
    if missing:
        pytest.skip(f"fixtures missing: {missing}")
    return True


def _run_ci(extra, env=None):
    cmd = [sys.executable, "-m", "perfxpert", "ci", *extra]
    full_env = os.environ.copy()
    current_pythonpath = full_env.get("PYTHONPATH")
    full_env["PYTHONPATH"] = (
        f"{PACKAGE_ROOT}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(PACKAGE_ROOT)
    )
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(PACKAGE_ROOT),
        env=full_env,
    )


def test_ci_cmd_rc0_on_improvement(_fixtures_exist) -> None:
    proc = _run_ci([str(BASELINE_DB), str(IMPROVED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] < 0


def test_ci_cmd_rc1_on_regression_above_threshold(_fixtures_exist) -> None:
    proc = _run_ci([str(BASELINE_DB), str(REGRESSED_DB), "--threshold", "2.0", "--format", "json"])
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] > 2.0
    assert "regressed" in proc.stderr.lower()


def test_ci_cmd_respects_env_threshold_override(_fixtures_exist) -> None:
    proc = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "50.0"},
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    proc2 = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "0.1"},
    )
    assert proc2.returncode == 1


def test_ci_cmd_cli_threshold_wins_over_env(_fixtures_exist) -> None:
    proc = _run_ci(
        [str(BASELINE_DB), str(REGRESSED_DB), "--threshold", "0.1", "--format", "json"],
        env={"PERFXPERT_CI_REGRESSION_THRESHOLD": "99.0"},
    )
    assert proc.returncode == 1


def test_ci_cmd_text_format_shows_regression_summary(_fixtures_exist) -> None:
    proc = _run_ci([str(BASELINE_DB), str(REGRESSED_DB), "--threshold", "2.0", "--format", "text"])
    assert proc.returncode == 1
    assert "wall delta" in proc.stdout.lower() or "runtime" in proc.stderr.lower()
    assert "regressed" in proc.stderr.lower() or "runtime regressed" in proc.stderr.lower()
