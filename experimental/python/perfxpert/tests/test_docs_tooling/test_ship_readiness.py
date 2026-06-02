#!/usr/bin/env python3
"""Ship-readiness gate for the docs-audit scanners (cycle-4).

This test runs all three docs scanners as subprocesses in --strict mode
and asserts each exits 0. It exists to prevent silent regression of the
docs gates post-merge — any new banned-string hit, dead internal link,
or non-executable code sample will turn CI red here.

All scanners live under ``experimental/python/perfxpert/scripts/``;
the lint script searches the relative path ``experimental/python/perfxpert``
while the python scanners take an explicit search root. We therefore
invoke everything from the rocm-systems repo root (parents[5] from this
test file, since the file now lives at
experimental/python/perfxpert/tests/test_docs_tooling/test_ship_readiness.py).
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCRIPTS = _REPO_ROOT / "experimental" / "python" / "perfxpert" / "scripts"
_LINT_SH = _SCRIPTS / "lint.sh"
_LINK_CHECKER = _SCRIPTS / "link-checker.py"
_TEST_SAMPLES = _SCRIPTS / "test-samples.py"
_PERFXPERT_ROOT = "experimental/python/perfxpert"


def _fmt(result: subprocess.CompletedProcess) -> str:
    """Compact error summary for pytest failure output."""
    return (
        f"exit={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_lint_sh_strict_exits_clean():
    """scripts/lint.sh --strict must exit 0 with no banned-string hits."""
    result = subprocess.run(
        ["bash", str(_LINT_SH), "--strict"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "scripts/lint.sh --strict reported violations:\n" + _fmt(result)
    )


def test_link_checker_strict_exits_clean():
    """scripts/link-checker.py --strict must report no dead internal links."""
    result = subprocess.run(
        [sys.executable, str(_LINK_CHECKER), "--strict", _PERFXPERT_ROOT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "scripts/link-checker.py --strict reported dead links:\n" + _fmt(result)
    )


def test_test_samples_strict_exits_clean():
    """scripts/test-samples.py --strict must report no non-executable samples."""
    result = subprocess.run(
        [sys.executable, str(_TEST_SAMPLES), "--strict", _PERFXPERT_ROOT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "scripts/test-samples.py --strict reported failing samples:\n"
        + _fmt(result)
    )
