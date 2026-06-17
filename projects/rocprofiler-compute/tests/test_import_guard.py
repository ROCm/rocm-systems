# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Test the ProfileModeImportGuard to ensure it works correctly."""

import os
import sys
from pathlib import Path

import pytest
from conftest import ProfileModeImportGuard


def test_import_guard_allows_stdlib_and_project():
    """Verify ProfileModeImportGuard allows stdlib and project imports."""
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
    test_packages = ["pandas", "yaml", "numpy"]

    for package in test_packages:
        # Should block the import
        with pytest.raises(ImportError, match="PROFILE MODE DEPENDENCY VIOLATION"):
            with ProfileModeImportGuard():
                __import__(package)

        # Verify error message mentions the forbidden package
        with pytest.raises(ImportError, match=package):
            with ProfileModeImportGuard():
                __import__(package)


def test_import_guard_catches_already_cached_package():
    """Guard must flag a forbidden import even when it is already cached."""
    # pandas is cached session-wide (conftest imports it via common.py); a
    # meta_path finder alone never sees a cached import.
    assert "pandas" in sys.modules
    with pytest.raises(ImportError, match="PROFILE MODE DEPENDENCY VIOLATION"):
        with ProfileModeImportGuard():
            __import__("pandas")
