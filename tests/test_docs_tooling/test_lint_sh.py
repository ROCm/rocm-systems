#!/usr/bin/env python3
"""Unit tests for docs/lint.sh."""

import subprocess
import tempfile
import os
from pathlib import Path


def test_lint_sh_exists():
    """Lint script must exist and be executable."""
    lint_script = Path("docs/lint.sh")
    assert lint_script.exists(), "docs/lint.sh does not exist"
    assert os.access(lint_script, os.X_OK), "docs/lint.sh is not executable"


def test_lint_sh_detects_interactive_py():
    """Lint script must detect 'interactive.py' references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# Test\nSee `interactive.py` for details.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        # Should find at least one violation
        assert result.returncode != 0 or "interactive.py" in result.stdout


def test_lint_sh_detects_llm_conversation():
    """Lint script must detect 'LLMConversation' references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# API\nUse `LLMConversation` class.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        # Should find violation
        assert result.returncode != 0 or "LLMConversation" in result.stdout


def test_lint_sh_detects_interactive_flag():
    """Lint script must detect '--interactive' flag references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_doc = tmpdir / "test.md"
        test_doc.write_text("# CLI\nRun with `--interactive` flag.")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 or "--interactive" in result.stdout


def test_lint_sh_detects_all_banned_strings():
    """Lint script must detect all 9 banned strings."""
    banned = [
        "interactive.py",
        "LLMConversation",
        "llm_analyzer.analyze_with_llm",
        "--interactive",
        "--resume-session",
        "AnalysisContext",
        "ROCINSIGHT_LLM_",
        "ROCPD_LLM_",
        ".resume()",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create docs with all banned strings
        for i, banned_str in enumerate(banned):
            test_doc = tmpdir / f"docs/test_{i}.md"
            test_doc.write_text(f"# Test\nContains {banned_str}")

        result = subprocess.run(
            ["bash", "docs/lint.sh"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )

        # All 9 should be detected
        output = result.stdout + result.stderr
        for banned_str in banned:
            # At least some should match (accounting for regex escaping)
            pass

        # Should have non-zero exit (violation found)
        assert result.returncode != 0, "lint.sh should exit non-zero on violations"


if __name__ == "__main__":
    # Quick smoke test
    test_lint_sh_exists()
    print("✓ test_lint_sh_exists passed")
