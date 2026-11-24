# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof-compute helpers."""

from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
from common import ROOT

script = Path(ROOT) / "src/rocprof-compute"
if not script.exists():
    script = Path(ROOT) / "rocprof-compute"
rocprof_compute = SourceFileLoader("rocprof-compute", str(script)).load_module()


@pytest.mark.parametrize(
    "local_ver, desired_ver, operator, expected",
    [
        ("2.5.5", "1.26.4", ">=", True),
        ("2.10.0", "2.9.0", ">=", True),
        ("1.26.3", "1.26.4", ">=", False),
        ("2.2.3", "2.2.3", "==", True),
        ("2.2.3", "2.2.2", "==", False),
        ("2.13.0+rocm7.15.0a20260728", "2.13.0", "==", True),
    ],
)
def test_check_version(local_ver, desired_ver, operator, expected):
    """Numeric tuple compare, not string compare."""
    assert rocprof_compute.check_version(local_ver, desired_ver, operator) is expected
