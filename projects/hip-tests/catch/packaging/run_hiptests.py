#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


# TODO(#3204): Re-enable tests once issues are resolved.
TEST_TO_IGNORE = {
    "gfx950-dcgpu": {
        "linux": [
            "Unit_hipHostRegister_AsyncApis",
            "Unit_hipMemsetDSync - uint32_t",
            "Unit_hipMemsetDASyncMulti - int8_t",
            "Unit_hipStreamValue_Wait_Blocking - uint32_t",
            "Unit_atomicExch_Positive_Same_Address_Compile_Time",
            "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - int",
            "Unit_hipHostRegister_Graphs",
            "Unit_hipManagedKeyword_SingleGpu",
            "Unit_hipMemsetSync",
            "Unit_hipMemset2DSync",
            "Unit_hipMemsetDASyncMulti - int16_t",
            "Unit_hipStreamValue_Wait_Blocking - uint64_t",
            "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - float",
            "Unit_hipMemsetDSync - int8_t",
            "Unit_hipMemset3DSync",
            "Unit_hipMemsetDASyncMulti - uint32_t",
            (
                "Unit_hipStreamValue_Write - "
                "TestParams<uint64_t, PtrType::DevicePtrToHost>"
            ),
            "Unit_hipHostRegister_ReferenceFromKernelandhipMemset - double",
            "Unit_hipGetProcAddress_MemoryApisRegisterUnReg",
            "Unit_hipMemsetDSync - int16_t",
            "Unit_hipMemsetASyncMulti",
            "Unit_hipHostAlloc_AllocateMoreThanAvailGPUMemory",
            (
                "Unit_hipStreamValue_Write - "
                "TestParams<uint32_t, PtrType::DevicePtrToHost>"
            ),
            # TODO(#4244): Flaky with compiler submodule update - subprocess aborted.
            "Unit_NonHost_Printf_loop",
            "Unit_NonHost_Printf_multiple_Threads",
            "Unit_NonHost_Printf_BufferAvailability",
        ]
    },
    "gfx94X-dcgpu": {
        "linux": [
            # TODO(#4244): Flaky with compiler submodule update - subprocess aborted.
            "Unit_NonHost_Printf_loop",
            "Unit_NonHost_Printf_multiple_Threads",
            "Unit_NonHost_Printf_BufferAvailability",
        ]
    },
    "gfx110X-all": {
        "windows": [
            "Unit_hipStreamValue_Wait_Blocking - uint64_t",
            "Unit_hipStreamValue_Wait_Blocking - uint32_t",
        ]
    },
}


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "catch_tests" and script_dir.parent.name == "hip":
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed hip catch test layout. "
        "Set ROCM_PATH explicitly."
    )


def is_asan() -> bool:
    explicit = os.getenv("HIP_TESTS_ASAN")
    if explicit:
        return explicit.lower() in ("1", "on", "true", "yes")
    return "asan" in os.getenv("ARTIFACT_GROUP", "").lower()


def get_asan_lib_path(rocm_path: Path) -> str:
    arch = platform.machine()
    clang_path = rocm_path / "lib" / "llvm" / "bin" / "clang++"
    cmd = [str(clang_path), f"--print-file-name=libclang_rt.asan-{arch}.so"]
    logging.info(f"++ Exec [{clang_path.parent}]$ {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def copy_dlls_exe_path(rocm_bin_dir: Path, catch_tests_path: Path) -> None:
    dlls_pattern = [
        "amdhip64*.dll",
        "amd_comgr*.dll",
        "hiprtc*.dll",
        "rocm_kpack*.dll",
    ]
    dlls_to_copy = []
    for pattern in dlls_pattern:
        dlls_to_copy.extend(rocm_bin_dir.glob(pattern))
    for dll in dlls_to_copy:
        try:
            shutil.copy(dll, catch_tests_path)
            logging.info(f"++ Copied: {dll} to {catch_tests_path}")
        except OSError as e:
            logging.info(f"++ Error copying {dll}: {e}")


def setup_env(env: Dict[str, str], rocm_path: Path, catch_tests_path: Path) -> None:
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()
    env["ROCM_PATH"] = str(rocm_path)
    if platform.system() == "Linux":
        hip_lib_path = rocm_path / "lib"
        logging.info(f"++ Setting LD_LIBRARY_PATH={hip_lib_path}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{hip_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(hip_lib_path)
        if is_asan():
            env["LD_PRELOAD"] = get_asan_lib_path(rocm_path)
            env["HSA_XNACK"] = "1"
    else:
        copy_dlls_exe_path(rocm_bin_dir, catch_tests_path)


def build_ctest_command(catch_tests_path: Path) -> List[str]:
    # Allow for more time in ASAN mode to run the tests.
    timeout = 1500 if is_asan() else 600
    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

    cmd = [
        "ctest",
        "--tests-information",
        f"{shard_index},,{total_shards}",
        "--test-dir",
        str(catch_tests_path),
        "--output-on-failure",
        "--timeout",
        str(timeout),
    ]

    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    os_type = platform.system().lower()
    if amdgpu_families in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[amdgpu_families]:
        ignored_tests = TEST_TO_IGNORE[amdgpu_families][os_type]
        cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

    return cmd


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    catch_tests_path = rocm_path / "share" / "hip" / "catch_tests"
    if not catch_tests_path.is_dir():
        raise FileNotFoundError(f"catch tests not found in {catch_tests_path}")

    env = os.environ.copy()
    setup_env(env, rocm_path, catch_tests_path)

    cmd = build_ctest_command(catch_tests_path)
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
