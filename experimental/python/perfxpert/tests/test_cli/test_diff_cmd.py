###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for ``perfxpert diff`` (Confluence row #7)."""

from __future__ import annotations

import json
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


def _run_perfxpert(args, check=False, **kw):
    """Invoke ``python -m perfxpert`` with ``args``; return CompletedProcess."""
    cmd = [sys.executable, "-m", "perfxpert", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kw)


def test_diff_cmd_rc0_on_regression(_fixtures_exist):
    """``perfxpert diff`` is informational — rc=0 even when regressed."""
    proc = _run_perfxpert(["diff", str(BASELINE_DB), str(REGRESSED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # JSON output parseable.
    data = json.loads(proc.stdout)
    assert data["schema_version"] == "0.3.1"
    assert "wall_delta_pct" in data


def test_diff_cmd_rc0_on_improvement(_fixtures_exist):
    proc = _run_perfxpert(["diff", str(BASELINE_DB), str(IMPROVED_DB), "--format", "json"])
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["wall_delta_pct"] < 0


def test_diff_cmd_webview_has_regression_table(tmp_path, _fixtures_exist):
    """Webview output contains a <table> with per-kernel rows."""
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
        ],
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    html_path = tmp_path / "diff_out.html"
    assert html_path.exists(), list(tmp_path.iterdir())
    html = html_path.read_text()
    assert "<table" in html
    # Kernel fixture names must appear.
    assert "matmul" in html or "conv2d" in html or "add" in html
    # And the standard shdr/sbody frame.
    assert '<section class="scard">' in html
    assert "dtable" in html
    # Regression should surface as red delta color somewhere.
    assert "#e84040" in html


def test_diff_cmd_missing_db_returns_rc2(tmp_path):
    """Missing DBs return rc=2 with a friendly message."""
    proc = _run_perfxpert(
        ["diff", str(tmp_path / "nope.db"), str(tmp_path / "also_nope.db"), "--format", "text"]
    )
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower()
