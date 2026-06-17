"""Verify --interactive and --resume-session are no longer registered argparse options."""

import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "perfxpert"] + args,
        capture_output=True, text=True, check=False,
    )


def test_interactive_flag_rejected():
    result = _run(["analyze", "--interactive", "-i", "/tmp/x.db"])
    # argparse returns 2 on unrecognized flag
    assert result.returncode == 2
    assert "--interactive" in (result.stderr + result.stdout)
    assert ("unrecognized" in result.stderr.lower()
            or "not allowed" in result.stderr.lower()
            or "invalid" in result.stderr.lower())


def test_resume_session_flag_rejected():
    result = _run(["analyze", "--resume-session", "/tmp/session.json"])
    assert result.returncode == 2
    assert "--resume-session" in (result.stderr + result.stdout)


def test_help_suggests_perfxpert_code_instead():
    """Users who type --interactive should get a hint to use perfxpert-code."""
    result = _run(["analyze", "--interactive"])
    assert result.returncode != 0
    hint = result.stderr + result.stdout
    # The error epilog should point to the new TUI
    assert "perfxpert-code" in hint or "interactive" in hint.lower()


def test_analyze_still_has_core_flags():
    """Regression guard: removing --interactive must NOT remove --format, -i, --llm, etc."""
    result = _run(["analyze", "--help"])
    assert result.returncode == 0
    for flag in ("--format", "-i", "--llm", "--prompt", "--source-dir", "-d", "-o"):
        assert flag in result.stdout, f"regression: {flag} disappeared from --help"
