#!/usr/bin/env python3
"""Ship-readiness gate for the docs-audit scanners (cycle-4).

This test runs all three docs scanners as subprocesses in --strict mode
and asserts each exits 0. It exists to prevent silent regression of the
docs gates post-merge — any new banned-string hit, dead internal link,
or non-executable code sample will turn CI red here.

All scanners live at the repo root under ``docs/``; the lint script
searches relative paths (``experimental/python/perfxpert`` + ``docs``)
while the python scanners take an explicit search root. We therefore
invoke everything from the repo root (parents[2] from this test file).
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINT_SH = _REPO_ROOT / "docs" / "lint.sh"
_LINK_CHECKER = _REPO_ROOT / "docs" / "link-checker.py"
_TEST_SAMPLES = _REPO_ROOT / "docs" / "test-samples.py"
_PERFXPERT_ROOT = "experimental/python/perfxpert"


def _fmt(result: subprocess.CompletedProcess) -> str:
    """Compact error summary for pytest failure output."""
    return (
        f"exit={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_lint_sh_strict_exits_clean():
    """docs/lint.sh --strict must exit 0 with no banned-string hits."""
    result = subprocess.run(
        ["bash", str(_LINT_SH), "--strict"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "docs/lint.sh --strict reported violations:\n" + _fmt(result)
    )


def test_link_checker_strict_exits_clean():
    """docs/link-checker.py --strict must report no dead internal links."""
    result = subprocess.run(
        [sys.executable, str(_LINK_CHECKER), "--strict", _PERFXPERT_ROOT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "docs/link-checker.py --strict reported dead links:\n" + _fmt(result)
    )


def test_test_samples_strict_exits_clean():
    """docs/test-samples.py --strict must report no non-executable samples."""
    result = subprocess.run(
        [sys.executable, str(_TEST_SAMPLES), "--strict", _PERFXPERT_ROOT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "docs/test-samples.py --strict reported failing samples:\n"
        + _fmt(result)
    )
