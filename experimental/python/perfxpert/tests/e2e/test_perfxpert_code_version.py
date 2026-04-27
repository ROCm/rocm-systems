"""E2E: `perfxpert-code --version` produces AMD-branded output."""

from importlib.metadata import version
import subprocess
import sys


def _run_perfxpert_code(*args: str) -> subprocess.CompletedProcess[str]:
    code = (
        "from perfxpert.cli.opencode_launcher import main; "
        f"raise SystemExit(main({list(args)!r}))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_version_flag_prints_amd_branding():
    result = _run_perfxpert_code("--version")
    assert result.returncode == 0
    assert "AMD" in result.stdout
    assert "ROCm PerfXpert" in result.stdout


def test_short_v_flag():
    result = _run_perfxpert_code("-V")
    assert result.returncode == 0
    assert "AMD" in result.stdout


def test_version_has_version_number():
    import re
    result = _run_perfxpert_code("--version")
    # something like "AMD ROCm PerfXpert 0.2.0 (opencode wrapper)"
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)
    assert version("perfxpert") in result.stdout
