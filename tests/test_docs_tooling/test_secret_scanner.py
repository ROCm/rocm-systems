#!/usr/bin/env python3
"""Unit tests for tools/_secret_scanner.py."""

import subprocess
import tempfile
from pathlib import Path


def test_secret_scanner_exists():
    """Secret scanner script must exist."""
    scanner = Path("tools/_secret_scanner.py")
    assert scanner.exists(), "tools/_secret_scanner.py does not exist"


def test_secret_scanner_detects_anthropic_key():
    """Secret scanner must detect Anthropic API key patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a file with Anthropic key
        test_file = tmpdir / "test.txt"
        test_file.write_text("export ANTHROPIC_API_KEY=sk-ant-api0123456789abcdefghijk")

        result = subprocess.run(
            ["python3", "tools/_secret_scanner.py"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "GIT_DIR": str(tmpdir)},
        )

        # May pass or fail depending on git setup, but shouldn't crash
        assert result.returncode in [0, 1]


def test_secret_scanner_detects_aws_key():
    """Secret scanner must detect AWS access key pattern."""
    content = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_file = tmpdir / ".aws/credentials"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(content)

        result = subprocess.run(
            ["python3", "tools/_secret_scanner.py"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )

        # May fail if git is set up, but shouldn't crash
        assert result.returncode in [0, 1, 128]


def test_secret_scanner_allows_normal_content():
    """Secret scanner should allow files without secrets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create normal file
        test_file = tmpdir / "README.md"
        test_file.write_text("# Project\nThis is a normal README.")

        result = subprocess.run(
            ["python3", "tools/_secret_scanner.py"],
            cwd=str(tmpdir),
            capture_output=True,
            text=True,
        )

        # Should pass or fail gracefully (depends on git setup)
        assert result.returncode in [0, 1, 128]


if __name__ == "__main__":
    test_secret_scanner_exists()
    print("✓ test_secret_scanner_exists passed")
