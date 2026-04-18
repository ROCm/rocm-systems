"""E2E: `perfxpert-code --version` produces AMD-branded output."""

import subprocess


def test_version_flag_prints_amd_branding():
    result = subprocess.run(
        ["perfxpert-code", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "AMD" in result.stdout
    assert "ROCm PerfXpert" in result.stdout


def test_short_v_flag():
    result = subprocess.run(
        ["perfxpert-code", "-V"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "AMD" in result.stdout


def test_version_has_version_number():
    import re
    result = subprocess.run(
        ["perfxpert-code", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    # something like "AMD ROCm PerfXpert 0.2.0 (opencode wrapper)"
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)
