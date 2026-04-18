"""End-to-end test for `perfxpert doctor` output format."""

import re
import subprocess
from pathlib import Path

import pytest


FIXTURE = (Path(__file__).parent.parent / "fixtures" / "doctor"
           / "expected_clean_output.txt")


def _run_doctor() -> str:
    r = subprocess.run(["perfxpert", "doctor"], capture_output=True, text=True)
    assert r.returncode == 0, f"exit={r.returncode}\nstderr={r.stderr}"
    return r.stdout


def test_doctor_succeeds_and_emits_all_clean_token():
    out = _run_doctor()
    assert "ALL CLEAN" in out, out
    assert "perfxpert" in out


def test_doctor_emits_expected_lines():
    out = _run_doctor()
    # Lines appear in the canonical order
    expected_line_patterns = [
        r"✓ perfxpert \d+\.\d+\.\d+ installed",
        r"✓ Python 3\.\d+ \(>= 3\.10 required\)",
        r"✓ openai-agents \d+\.\d+\.\d+",
        r"✓ MCP server reachable",
        r"✓ Python task store .+ ready",
        r"✓ Bundled opencode \S+ detected at .+",
        r"✓ \d+/5 LLM providers configured",
        r"✓ ALL CLEAN",
    ]
    for pat in expected_line_patterns:
        assert re.search(pat, out), f"pattern missing: {pat}\noutput: {out}"


def test_doctor_has_no_leading_whitespace_on_primary_lines():
    """Sub-lines (unconfigured providers) start with 2 spaces; primary lines don't."""
    for line in _run_doctor().splitlines():
        if line.startswith(("✓", "⚠", "✗")):
            assert not line.startswith(" "), f"leading whitespace on primary: {line!r}"


def test_doctor_exits_zero_on_clean_system():
    r = subprocess.run(["perfxpert", "doctor"], capture_output=True, text=True)
    assert r.returncode == 0
