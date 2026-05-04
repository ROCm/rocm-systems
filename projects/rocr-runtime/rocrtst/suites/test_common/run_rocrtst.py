#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path


logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
ROCM_PATH = Path(os.environ.get("ROCM_PATH", SCRIPT_DIR.parent.parent)).resolve()
ROCM_BIN_DIR = Path(os.environ.get("ROCM_BIN_DIR", ROCM_PATH / "bin")).resolve()
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
OS_TYPE = platform.system().lower()

# GTest sharding.
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")
env = os.environ.copy()
# GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
env["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
env["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# TODO(#3851): Excluded tests (flaky or disabled in CI).
TEST_TO_IGNORE = {
    "gfx90a": {
        "linux": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx94X-dcgpu": {
        "linux": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx950-dcgpu": {
        "linux": [
            "rocrtstFunc.GpuCoreDump_DefaultPattern",
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx110X-all": {
        "windows": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx1151": {
        "windows": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
}

# If quick tests are enabled, run quick tests only. Otherwise, run the full suite.
QUICK_TESTS = [
    "rocrtst.Test_Example",
    "rocrtstFunc.MemoryAccessTests",
    "rocrtstFunc.GroupMemoryAllocationTest",
    "rocrtstFunc.MemoryAllocateAndFreeTest",
    "rocrtstFunc.Memory_Alignment_Test",
    "rocrtstFunc.Concurrent_Init_Test",
    "rocrtstFunc.Concurrent_Init_Shutdown_Test",
    "rocrtstFunc.Reference_Count",
    "rocrtstFunc.Signal_Create_Concurrently",
    "rocrtstFunc.Signal_Destroy_Concurrently",
    "rocrtstFunc.IPC",
    "rocrtstFunc.AgentProp_UUID",
    "rocrtstFunc.Deallocation_Notifier_Test",
    "rocrtstFunc.Memory_Atomic_Add_Test",
    "rocrtstFunc.Memory_Atomic_Xchg_Test",
]

exclude_filter = "-"
if AMDGPU_FAMILIES in TEST_TO_IGNORE and OS_TYPE in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][OS_TYPE]
    exclude_filter += ":".join(ignored_tests)

test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    env["GTEST_FILTER"] = ":".join(QUICK_TESTS) + ":" + exclude_filter
else:
    env["GTEST_FILTER"] = exclude_filter

cmd = ["./rocrtst64"]
logging.info(f"++ Exec [{ROCM_BIN_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=ROCM_BIN_DIR, check=True, env=env)
