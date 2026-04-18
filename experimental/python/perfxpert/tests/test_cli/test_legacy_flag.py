"""PERFXPERT_LEGACY=1 safety-net behavior."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _perfxpert_cli() -> list[str]:
    """Build a CLI command that invokes perfxpert via python -m (in-tree)."""
    return [sys.executable, "-m", "perfxpert"]


def test_perfxpert_analyze_help_does_not_mention_removed_flags():
    """--interactive and --resume-session should be absent from --help after Phase 6."""
    result = subprocess.run(
        _perfxpert_cli() + ["analyze", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    help_text = result.stdout
    assert "--interactive" not in help_text
    assert "--resume-session" not in help_text


def test_PERFXPERT_LEGACY_1_prints_warning_on_cli(tmp_path):
    """Running `perfxpert analyze` with PERFXPERT_LEGACY=1 must emit a stderr warning."""
    db = tmp_path / "empty.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)  # minimal valid-looking stub

    env = os.environ.copy()
    env["PERFXPERT_LEGACY"] = "1"

    result = subprocess.run(
        _perfxpert_cli() + ["analyze", "-i", str(db)],
        capture_output=True, text=True, check=False, env=env,
    )
    combined = result.stderr + result.stdout
    assert "DEPRECATED" in combined.upper() or "deprecation" in combined.lower()
    assert "PERFXPERT_LEGACY" in combined


def test_doctor_reports_agentic_mode_by_default():
    """`perfxpert doctor` prints 'Mode: agentic' when no legacy flag is set."""
    env = os.environ.copy()
    env.pop("PERFXPERT_LEGACY", None)

    result = subprocess.run(
        _perfxpert_cli() + ["doctor"],
        capture_output=True, text=True, check=False, env=env,
    )
    out = result.stdout + result.stderr
    # must mention agentic mode somewhere
    assert "Mode: agentic" in out or "agentic path (default)" in out


def test_doctor_reports_legacy_mode_when_flag_set():
    """`perfxpert doctor` prints 'Mode: legacy' when PERFXPERT_LEGACY=1."""
    env = os.environ.copy()
    env["PERFXPERT_LEGACY"] = "1"

    result = subprocess.run(
        _perfxpert_cli() + ["doctor"],
        capture_output=True, text=True, check=False, env=env,
    )
    out = result.stdout + result.stderr
    assert "Mode: legacy" in out
    assert "DEPRECATED" in out.upper() or "deprecation" in out.lower()
