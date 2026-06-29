# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Validate that production YAML config files have properly formatted metric equations.

This is a CI gate: if any YAML config file has equations that don't match
the canonical format (operator spacing, constant factoring, parentheses
minimization), the test fails.  Run ``tools/format_yaml.py --fix`` to
auto-fix any failures.
"""

import subprocess
import sys
from pathlib import Path

import common
import pytest

FORMAT_YAML = Path(common.ROOT) / "tools/format_yaml.py"
ANALYSIS_CONFIGS = Path(common.SRC) / "rocprof_compute_soc/analysis_configs"
TUI_CONFIGS = Path(common.SRC) / "rocprof_compute_tui/utils"

YAML_DIRS = [ANALYSIS_CONFIGS, TUI_CONFIGS]


def collect_yaml_files():
    files = []
    for yaml_dir in YAML_DIRS:
        if yaml_dir.is_dir():
            files.extend(sorted(yaml_dir.rglob("*.yaml")))
    return files


yaml_files = collect_yaml_files()
assert yaml_files, "No YAML config files found — check paths"


@pytest.mark.parametrize(
    "yaml_path",
    yaml_files,
    ids=[str(p.relative_to(common.ROOT)) for p in yaml_files],
)
def test_yaml_equations_are_formatted(yaml_path: Path):
    """Each production YAML config must have properly formatted equations."""
    result = subprocess.run(
        [sys.executable, str(FORMAT_YAML), "--diff", str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{yaml_path.relative_to(common.ROOT)} has formatting issues:\n"
        + result.stdout
        + "\n\nRun: tools/format_yaml.py --fix <file>"
    )
