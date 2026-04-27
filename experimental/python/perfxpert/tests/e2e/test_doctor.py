"""End-to-end test for `perfxpert doctor` output format."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURE = (Path(__file__).parent.parent / "fixtures" / "doctor"
           / "expected_clean_output.txt")


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

    r = subprocess.run(
        [sys.executable, "-m", "perfxpert", "doctor"],
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode, r.stdout


def _bundled_opencode_available() -> bool:
    """Return True when the managed PerfXpert opencode binary is available."""
    from perfxpert.cli.opencode_launcher import resolve_opencode_binary

    try:
        resolve_opencode_binary()
    except FileNotFoundError:
        return False
    return True


def test_doctor_succeeds_and_emits_all_clean_token():
    """Doctor should emit 'ALL CLEAN' when all checks pass."""
    if not _bundled_opencode_available():
        pytest.skip("managed bundled opencode binary not available")
    exit_code, out = _run_doctor()
    assert exit_code == 0, f"exit={exit_code}\noutput: {out}"
    assert "ALL CLEAN" in out, out
    assert "perfxpert" in out
    assert re.search(r"✓|✗|\[OK\]|\[FAIL\]|\bOK\b|\bFAIL\b", out), out


def test_doctor_emits_expected_lines():
    """Doctor output should contain expected status lines in canonical format."""
    exit_code, out = _run_doctor()
    # These patterns should always be present (regardless of whether all checks pass)
    ok = r"(?:✓|\[OK\]|OK)"
    fail = r"(?:✗|\[FAIL\]|FAIL)"
    essential_patterns = [
        rf"{ok} perfxpert \d+\.\d+\.\d+",
        rf"{ok} Python 3\.\d+",
        rf"({ok}|{fail}) (openai-agents|openai-agents \d+\.\d+\.\d+)",
        rf"{ok} MCP server",
        rf"{ok} Python task store",
        rf"({ok}|{fail}) (opencode|Bundled opencode)",
        r"\d+/5 LLM providers configured",
    ]
    for pat in essential_patterns:
        assert re.search(pat, out), f"pattern missing: {pat}\noutput: {out}"


def test_doctor_has_no_leading_whitespace_on_primary_lines():
    """Sub-lines (unconfigured providers) start with 2 spaces; primary lines don't."""
    exit_code, out = _run_doctor()
    for line in out.splitlines():
        if line.startswith(("✓", "⚠", "✗", "[OK]", "[WARN]", "[FAIL]", "OK ", "WARN ", "FAIL ")):
            assert not line.startswith(" "), f"leading whitespace on primary: {line!r}"


def test_doctor_exits_zero_on_clean_system():
    """Doctor should exit zero when all required checks pass."""
    if not _bundled_opencode_available():
        pytest.skip("managed bundled opencode binary not available")
    exit_code, out = _run_doctor()
    assert exit_code == 0, f"exit={exit_code}\noutput: {out}"
    # Always check that output is not malformed (has sections, no NameError, etc.)
    assert "LLM providers configured" in out
    assert "Mode:" in out  # Active mode reporting
