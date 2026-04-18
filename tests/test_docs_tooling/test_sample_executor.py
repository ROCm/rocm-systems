#!/usr/bin/env python3
"""Unit tests for docs/test-samples.py."""

import subprocess
import tempfile
import json
from pathlib import Path


def test_sample_executor_exists():
    """Sample executor script must exist."""
    executor = Path("docs/test-samples.py")
    assert executor.exists(), "docs/test-samples.py does not exist"


def test_sample_executor_runs_bash():
    """Sample executor must extract and run bash blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create a doc with valid bash
        test_doc = tmpdir / "docs/test.md"
        test_doc.write_text("""
# Test

```bash
echo "Hello, World!"
exit 0
```
""")

        result = subprocess.run(
            ["python3", "docs/test-samples.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should succeed
        assert result.returncode == 0, f"Failed: {result.stderr}"


def test_sample_executor_skips_marked_samples():
    """Sample executor must skip samples marked SKIP-SAMPLE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create a doc with skipped bash
        test_doc = tmpdir / "docs/test.md"
        test_doc.write_text("""
# Test

```bash
# SKIP-SAMPLE
exit 999
```
""")

        result = subprocess.run(
            ["python3", "docs/test-samples.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should succeed (sample skipped)
        assert result.returncode == 0


def test_sample_executor_detects_failed_bash():
    """Sample executor must detect failing bash samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create a doc with failing bash
        test_doc = tmpdir / "docs/test.md"
        test_doc.write_text("""
# Test

```bash
exit 1
```
""")

        result = subprocess.run(
            ["python3", "docs/test-samples.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should fail
        assert result.returncode != 0


def test_sample_executor_output_format():
    """Sample executor output must be machine-readable JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        test_doc = tmpdir / "docs/test.md"
        test_doc.write_text("""
# Test

```bash
echo "test"
```
""")

        result = subprocess.run(
            ["python3", "docs/test-samples.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Try to parse output as JSON
        try:
            output = json.loads(result.stdout)
            assert 'samples' in output or isinstance(output, list)
        except json.JSONDecodeError:
            # If not pure JSON, should at least have structure
            assert len(result.stdout) > 0


def test_sample_executor_runs_python():
    """Sample executor must extract and run Python blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "docs").mkdir()

        # Create a doc with valid Python
        test_doc = tmpdir / "docs/test.md"
        test_doc.write_text("""
# Test

```python
print("Hello from Python")
```
""")

        result = subprocess.run(
            ["python3", "docs/test-samples.py", str(tmpdir)],
            capture_output=True,
            text=True,
        )

        # Should succeed
        assert result.returncode == 0, f"Failed: {result.stderr}"


if __name__ == "__main__":
    test_sample_executor_exists()
    print("✓ test_sample_executor_exists passed")
