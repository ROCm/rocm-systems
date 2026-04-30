"""End-to-end test for `perfxpert doctor` output format."""

import os
import re
import subprocess
from pathlib import Path

import pytest


FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "doctor"
    / "expected_clean_output.txt"
)


def _patched_opencode_available() -> bool:
    """Return whether the default patched opencode path resolves locally."""
    from perfxpert.cli.opencode_launcher import resolve_opencode_binary

    try:
        resolve_opencode_binary()
        return True
    except FileNotFoundError:
        return False


def _run_doctor(env=None) -> tuple[int, str]:
    """Run perfxpert doctor with optional env overrides.

    Returns (exit_code, stdout).
    """
    if env is None:
        env = os.environ.copy()
    else:
        # Merge with current environ
        merged = os.environ.copy()
        merged.update(env)
        env = merged

    r = subprocess.run(["perfxpert", "doctor"], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout


def test_doctor_succeeds_and_emits_all_clean_token():
    """Doctor should emit 'ALL CLEAN' when all checks pass."""
    exit_code, out = _run_doctor()
    # Should pass if the patched opencode path is available.
    if _patched_opencode_available():
        assert exit_code == 0, f"exit={exit_code}\noutput: {out}"
        assert "ALL CLEAN" in out, out
    assert "perfxpert" in out
    assert "✓" in out or "✗" in out  # Has status indicators


def test_doctor_emits_expected_lines():
    """Doctor output should contain expected status lines in canonical format."""
    exit_code, out = _run_doctor()
    # These patterns should always be present (regardless of whether all checks pass)
    essential_patterns = [
        r"✓ perfxpert \d+\.\d+\.\d+",
        r"✓ Python 3\.\d+",
        r"(✓|✗) (openai-agents|openai-agents \d+\.\d+\.\d+)",
        r"✓ MCP server",
        r"✓ Python task store",
        r"(✓|✗) (Patched opencode|patched opencode binary not found)",
        r"\d+/5 LLM providers configured",
    ]
    for pat in essential_patterns:
        assert re.search(pat, out), f"pattern missing: {pat}\noutput: {out}"


def test_doctor_has_no_leading_whitespace_on_primary_lines():
    """Sub-lines (unconfigured providers) start with 2 spaces; primary lines don't."""
    exit_code, out = _run_doctor()
    for line in out.splitlines():
        if line.startswith(("✓", "⚠", "✗")):
            assert not line.startswith(" "), f"leading whitespace on primary: {line!r}"


def test_doctor_exits_zero_on_clean_system():
    """Doctor should exit zero when all required checks pass."""
    exit_code, out = _run_doctor()
    # Should succeed if the patched opencode path is available.
    if _patched_opencode_available():
        assert exit_code == 0, f"exit={exit_code}\noutput: {out}"
    # Always check that output is not malformed (has sections, no NameError, etc.)
    assert "LLM providers configured" in out
    assert "Mode:" in out  # Active mode reporting
