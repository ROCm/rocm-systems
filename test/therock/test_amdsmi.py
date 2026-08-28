#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AMDSMI Test Runner (Manual Execution Only)

This script is NOT part of automated CI runs.

`amdsmitst` requires GPU device access (/dev/kfd, /dev/dri), elevated
permissions, and execution on a ROCm-enabled system. GitHub-hosted CI
environments do not expose these capabilities, so this script must be run
manually by developers inside a privileged ROCm environment or container.

Usage:
    python test_amdsmi.py

===============================================================================
"""

import pytest

pytestmark = pytest.mark.skip("Manual execution only — requires GPU device access")
import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

AMDSMITST_BIN = (
    THEROCK_DIR / "build" / "share" / "amd_smi" / "tests" / "amdsmitst"
).resolve()

# -----------------------------
# GTest sharding
# -----------------------------
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")
env = os.environ.copy()

# Convert to 0-based index for GTest
env["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
env["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# -----------------------------
# Test filtering
# -----------------------------
# If quick mode is enabled, run the minimal test suite (C++ unit suite).
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    logging.info("Running quick (C++ unit tests only) for amdsmitst")
    test_filter = ["--gtest_filter=*Unit*"]
else:
    # Full test mode: run broad suites and explicitly exclude known failing tests
    logging.info("Running full amdsmitst test suite (include + exclude filter)")

    include_tests = [
        "*FunctionalReadOnly.*",
        "*FunctionalReadWrite.*",
        "*Unit*",
    ]

    # Explicitly exclude known failing tests, so that we can catch regressions in the future
    # ie. Don't silently skip or ignore runnable tests not shown in the exclude_tests list
    exclude_tests = [
        "GpuFunctionalReadOnly.TempRead",
        "GpuFunctionalReadOnly.TestFrequenciesRead",
        "GpuFunctionalReadWrite.TestPowerReadWrite",
    ]

    gtest_filter = f"{':'.join(include_tests)}:-{':'.join(exclude_tests)}"
    test_filter = [f"--gtest_filter={gtest_filter}"]

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDSMITST_BIN)] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

# -----------------------------
# Run tests
# -----------------------------
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=env,
    check=True,
)
