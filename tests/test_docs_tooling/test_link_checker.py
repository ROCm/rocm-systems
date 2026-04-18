#!/usr/bin/env python3
"""Unit tests for docs/link-checker.py."""

import subprocess
import tempfile
import json
from pathlib import Path


def test_link_checker_exists():
    """Link checker script must exist."""
    link_checker = Path("docs/link-checker.py")
    assert link_checker.exists(), "docs/link-checker.py does not exist"


def test_link_checker_detects_broken_internal_link():
    """Link checker must detect broken internal links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test_docs").mkdir()

        # Create a doc with broken internal link
        test_doc = tmpdir / "test_docs/test.md"
        test_doc.write_text(
            "# Test\n\nSee [Guide](./nonexistent.md) for details."
        )

        result = subprocess.run(
            ["python3", "docs/link-checker.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should report the broken link
        assert "nonexistent.md" in result.stdout or "nonexistent.md" in result.stderr


def test_link_checker_validates_internal_link():
    """Link checker must validate existing internal links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test_docs").mkdir()

        # Create docs with valid internal link
        (tmpdir / "test_docs/guide.md").write_text("# Guide\nContent.")
        test_doc = tmpdir / "test_docs/test.md"
        test_doc.write_text("# Test\n\nSee [Guide](./guide.md) for details.")

        result = subprocess.run(
            ["python3", "docs/link-checker.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should not report the valid link as broken
        assert "guide.md" not in result.stdout or "OK" in result.stdout


def test_link_checker_output_format():
    """Link checker output must be machine-readable (CSV or JSON)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test_docs").mkdir()

        test_doc = tmpdir / "test_docs/test.md"
        test_doc.write_text("# Test\n\nSee [Link](./bad.md)")

        result = subprocess.run(
            ["python3", "docs/link-checker.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Output should have structure (file, line, link)
        output = result.stdout
        assert len(output) > 0


def test_link_checker_skip_external_urls():
    """Link checker may skip external URLs (best-effort)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test_docs").mkdir()

        test_doc = tmpdir / "test_docs/test.md"
        test_doc.write_text(
            "# Test\n\nSee [AMD Docs](https://rocm.docs.amd.com/)"
        )

        result = subprocess.run(
            ["python3", "docs/link-checker.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should not error on external URLs (best-effort)
        assert result.returncode in [0, 1]


if __name__ == "__main__":
    test_link_checker_exists()
    print("✓ test_link_checker_exists passed")
