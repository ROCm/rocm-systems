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


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        bin_dir = candidate / "bin"
        if (bin_dir / "rocrtst64").is_file() or (bin_dir / "rocrtst64.exe").is_file():
            return candidate
    if script_dir.name == "rocrtst" and script_dir.parent.name == "share":
        return script_dir.parent.parent
    return script_dir.parent.parent


def build_gtest_filter(
    amdgpu_families: str | None, os_type: str, test_type: str
) -> str:
    exclude_filter = ""
    if amdgpu_families in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[amdgpu_families]:
        ignored_tests = TEST_TO_IGNORE[amdgpu_families][os_type]
        exclude_filter = "-" + ":".join(ignored_tests)

    if test_type == "quick":
        return ":".join(QUICK_TESTS) + exclude_filter
    return exclude_filter


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path = Path(
        os.environ.get("ROCM_PATH", derive_rocm_path(script_dir))
    ).resolve()
    rocm_bin_dir = Path(os.environ.get("ROCM_BIN_DIR", rocm_path / "bin")).resolve()

    env = os.environ.copy()
    # GitHub Actions shard arrays are 1-indexed; GTest shard indexes are 0-indexed.
    env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
    env["GTEST_TOTAL_SHARDS"] = os.getenv("TOTAL_SHARDS", "1")

    gtest_filter = build_gtest_filter(
        os.getenv("AMDGPU_FAMILIES"),
        platform.system().lower(),
        os.getenv("TEST_TYPE", "full"),
    )
    if gtest_filter:
        env["GTEST_FILTER"] = gtest_filter
    else:
        env.pop("GTEST_FILTER", None)

    cmd = ["./rocrtst64"]
    logging.info(f"++ Exec [{rocm_bin_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_bin_dir, check=True, env=env)


if __name__ == "__main__":
    main()
