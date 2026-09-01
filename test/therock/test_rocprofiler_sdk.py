# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Base Paths
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent

# LIB Paths
THEROCK_LIB_PATH = THEROCK_PATH / "lib"
THEROCK_SYSDEPS_PATH = THEROCK_LIB_PATH / "rocm_sysdeps"
THEROCK_SYSDEPS_LIB_PATH = THEROCK_SYSDEPS_PATH / "lib"

# LLVM Paths
THEROCK_LLVM_BIN_PATH = THEROCK_PATH / "llvm" / "bin"
THEROCK_CLANG_PATH = THEROCK_LLVM_BIN_PATH / "amdclang"
THEROCK_CLANG_PLUS_PATH = THEROCK_LLVM_BIN_PATH / "amdclang++"

# SDK Paths
ROCPROFILER_SDK_PATH = THEROCK_PATH / "share" / "rocprofiler-sdk"
ROCPROFILER_SDK_TESTS_PATH = ROCPROFILER_SDK_PATH / "tests"

# Architectures the PC sampling suites are exercised on. Deliberately narrower than
# rocprofiler_sdk_pc_sampling_disabled() in the SDK's cmake utilities, which admits every
# architecture the feature nominally supports.
PC_SAMPLING_ARCHITECTURES = re.compile(r"^(gfx942|gfx950|gfx1250)$")

logging.basicConfig(level=logging.INFO)
environ_vars = os.environ.copy()


def gpu_architecture():
    """Returns the gfx architecture of the first GPU agent, or "unknown" if it cannot be
    determined."""
    rocminfo = THEROCK_BIN_PATH / "rocminfo"
    if not rocminfo.exists():
        logging.warning(f"{rocminfo} not found")
        return "unknown"

    result = subprocess.run(
        [str(rocminfo)], capture_output=True, text=True, env=environ_vars
    )
    if result.returncode != 0:
        logging.warning(f"{rocminfo} returned {result.returncode}:\n{result.stderr}")
        return "unknown"

    # The CPU agents listed ahead of the GPUs carry no gfx name, so the first match is the
    # architecture of the first GPU agent.
    architecture = re.search(r"gfx[0-9a-fA-F]+", result.stdout)
    return architecture.group() if architecture else "unknown"


def setup_pc_sampling_env():
    """PC sampling is a beta feature that has to be opted into, and the suites using it
    report as skipped without the opt-in. The variable grants every rocprofiler-sdk process
    access to the feature, so only opt in on architectures that support PC sampling, and
    leave a setting made by the caller alone."""
    if "ROCPROFILER_PC_SAMPLING_BETA_ENABLED" in environ_vars:
        return

    architecture = gpu_architecture()
    if not PC_SAMPLING_ARCHITECTURES.match(architecture):
        logging.info(f"PC sampling beta feature not opted into for {architecture}")
        return

    logging.info(f"PC sampling beta feature opted into for {architecture}")
    environ_vars["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] = "1"


def setup_env():
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
    environ_vars["HIP_PATH"] = str(THEROCK_PATH)
    environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"

    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "").split(":")
    environ_vars["LD_LIBRARY_PATH"] = ":".join(
        [f"{THEROCK_LIB_PATH}", f"{THEROCK_SYSDEPS_LIB_PATH}"] + old_ld_lib_path
    )

    # rocminfo needs the library path above, so this has to come last.
    setup_pc_sampling_env()


def cmake_config():
    cmake_config_cmd = [
        "cmake",
        "-B",
        "build",
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_SYSDEPS_PATH}",
        f"-DCMAKE_HIP_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DCMAKE_C_COMPILER={THEROCK_CLANG_PATH}",
        f"-DCMAKE_CXX_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

    logging.info(
        f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(cmake_config_cmd)}"
    )
    subprocess.run(
        cmake_config_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


# SDK requires test binaries to be built on the gfx architecture being tested on
# Certain tests are enabled/disabled based on the GPU architecture.
# Ensuring that these tests build properly against an install is also part of the overall test coverage for SDK (emulates tool developers building tools with rocprofiler-sdk)
def cmake_build():
    cmake_build_cmd = [
        "cmake",
        "--build",
        "build",
        "--parallel",
        "8",
    ]

    logging.info(
        f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(cmake_build_cmd)}"
    )
    subprocess.run(
        cmake_build_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


def execute_tests():
    ctest_cmd = [
        "ctest",
        "--test-dir",
        "build",
        "--parallel",
        "8",
        "--output-on-failure",
    ]

    logging.info(f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(ctest_cmd)}")
    subprocess.run(
        ctest_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


if __name__ == "__main__":
    setup_env()
    cmake_config()
    cmake_build()
    execute_tests()
