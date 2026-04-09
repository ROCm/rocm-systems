##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

"""Test the ProfileModeImportGuard to ensure it works correctly."""

import os
import sys
from pathlib import Path

import pytest


def test_import_guard_allows_stdlib_and_project():
    """Verify ProfileModeImportGuard allows stdlib and project imports."""
    from conftest import ProfileModeImportGuard

    with ProfileModeImportGuard():
        # 1. Stdlib imports (must succeed - these are always available)
        import argparse
        import json
        import pathlib

        assert json is not None
        assert argparse is not None
        assert pathlib is not None

        # 2. Project modules (must succeed - sys.path includes src/)
        import config
        import utils

        assert utils is not None
        assert config is not None


def test_import_guard_allows_rocm_modules():
    """Verify ProfileModeImportGuard allows ROCm system libraries."""
    from conftest import ProfileModeImportGuard

    # Add amdsmi to sys.path (same as profile code does)
    amdsmi_path = Path(os.getenv("ROCM_PATH", "/opt/rocm")) / "share/amd_smi"
    if not amdsmi_path.exists():
        pytest.skip("ROCm not installed - skipping ROCm module import test")

    sys.path.insert(0, str(amdsmi_path))

    # Check if amdsmi module exists
    try:
        import amdsmi
    except ImportError:
        pytest.skip("amdsmi module not found - skipping ROCm module import test")

    # Clear from cache to ensure guard is actually tested
    if "amdsmi" in sys.modules:
        del sys.modules["amdsmi"]

    # Now test that the guard allows it
    with ProfileModeImportGuard():
        import amdsmi

        assert amdsmi is not None


def test_import_guard_blocks_non_stdlib():
    """Verify ProfileModeImportGuard blocks non-stdlib imports."""
    from conftest import ProfileModeImportGuard

    if sys.version_info < (3, 10):
        pytest.skip(
            "ProfileModeImportGuard requires Python 3.10+ (sys.stdlib_module_names)"
        )

    # Test blocking non-stdlib imports
    # Note: Must clear from sys.modules first since import hooks only
    # run for uncached modules
    test_packages = ["pandas", "yaml", "numpy"]

    for package in test_packages:
        # Clear from cache if present
        if package in sys.modules:
            del sys.modules[package]

        # Should block the import
        with pytest.raises(ImportError, match="PROFILE MODE DEPENDENCY VIOLATION"):
            with ProfileModeImportGuard():
                __import__(package)

        # Verify error message mentions the forbidden package
        if package in sys.modules:
            del sys.modules[package]
        with pytest.raises(ImportError, match=package):
            with ProfileModeImportGuard():
                __import__(package)
