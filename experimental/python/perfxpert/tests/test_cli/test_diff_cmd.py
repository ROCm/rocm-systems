"""Tests for ``perfxpert diff``."""

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


def _run_perfxpert(args, check=False, **kwargs):
    cmd = [sys.executable, "-m", "perfxpert", *args]
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PACKAGE_ROOT}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(PACKAGE_ROOT)
    )
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        cwd=str(PACKAGE_ROOT),
        env=env,
        **kwargs,
    )


def test_diff_cmd_rc0_on_regression(_fixtures_exist) -> None:
    proc = _run_perfxpert(["diff", str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["schema_version"] == "0.3.1"
    assert "wall_delta_pct" in data


def test_diff_cmd_rc0_on_improvement(_fixtures_exist) -> None:
    proc = _run_perfxpert(["diff", str(BASELINE_DB), str(IMPROVED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] < 0


def test_diff_cmd_webview_writes_report(tmp_path, _fixtures_exist) -> None:
    proc = _run_perfxpert(
        [
            "diff",
            str(BASELINE_DB),
            str(REGRESSED_DB),
            "--format",
            "webview",
            "-d",
            str(tmp_path),
            "-o",
            "diff_out",
        ]
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    html_path = tmp_path / "diff_out.html"
    assert html_path.exists(), list(tmp_path.iterdir())
    html = html_path.read_text()
    assert "<table" in html
    assert '<section class="scard">' in html
    assert "dtable" in html


def test_diff_cmd_missing_db_returns_rc2(tmp_path) -> None:
    proc = _run_perfxpert(
        ["diff", str(tmp_path / "nope.db"), str(tmp_path / "also_nope.db"), "--format", "text"]
    )
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower()
