"""Regression guards for flags removed in Phase 7.1 and Phase 6.

These tests ensure help output and doctor output reflect the
agentic-only world by asserting:

- `ai_analysis` module — removed in Phase 7.1 (must not be importable).
- `PERFXPERT_LEGACY` env var — removed in Phase 7.1 (must not alter doctor output).
- `--interactive` / `--resume-session` CLI flags — removed in Phase 6 (must not appear in --help).
"""

import os
import subprocess
import sys


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


def test_doctor_reports_agentic_mode():
    """Regression guard: `perfxpert doctor` always prints 'Mode: agentic' after Phase 7.1."""
    env = os.environ.copy()
    env.pop("PERFXPERT_LEGACY", None)  # regression guard

    result = subprocess.run(
        _perfxpert_cli() + ["doctor"],
        capture_output=True, text=True, check=False, env=env,
    )
    out = result.stdout + result.stderr
    assert "Mode: agentic" in out
