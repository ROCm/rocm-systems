#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Tests for tools/counter_grouping_inspector.py."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from utils.mi_gpu_spec import mi_gpu_specs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTER_GROUPING_INSPECTOR = PROJECT_ROOT / "tools" / "counter_grouping_inspector.py"
SUMMARY_RE = re.compile(r"Summary: (\d+) bucket\(s\), (\d+) counter assignment\(s\)")


def run_inspector(arch: str, timeout: int) -> subprocess.CompletedProcess:
    """Run the inspector for arch from the project directory."""
    return subprocess.run(
        [sys.executable, str(COUNTER_GROUPING_INSPECTOR), "--arch", arch],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
    )


@pytest.mark.parametrize("arch", sorted(mi_gpu_specs.get_gpu_series_dict()))
def test_groups_counters_of_supported_arch(arch):
    result = run_inspector(arch, timeout=120)
    assert result.returncode == 0, result.stderr[:500]
    summary = SUMMARY_RE.search(result.stdout)
    assert summary, result.stdout[-500:]
    buckets, counter_assignments = map(int, summary.groups())
    assert buckets > 0
    assert counter_assignments > 0


def test_rejects_unknown_arch():
    result = run_inspector("__not_a_valid_arch__", timeout=60)
    assert result.returncode == 2
    assert "invalid choice" in (result.stderr + result.stdout).lower()
