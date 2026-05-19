# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
if THEROCK_BIN_DIR_STR is None:
    logging.info(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)
THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR)
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()
THEROCK_TEST_DIR = THEROCK_DIR / "build"

ROCJPEG_TEST_PATH = str(THEROCK_BIN_DIR.resolve().parent / "share" / "rocjpeg" / "test")
if not os.path.isdir(ROCJPEG_TEST_PATH):
    logging.info(f"++ Error: rocjpeg tests not found in {ROCJPEG_TEST_PATH}")
    sys.exit(1)
else:
    logging.info(f"++ INFO: rocjpeg tests found in {ROCJPEG_TEST_PATH}")
env = os.environ.copy()


def setup_env(env):
    ROCM_PATH = THEROCK_BIN_DIR.resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)
    logging.info(f"++ rocjpeg setting ROCM_PATH={ROCM_PATH}")
    if platform.system() == "Linux":
        hip_lib_path = THEROCK_BIN_DIR.resolve().parent / "lib"
        logging.info(f"++ rocjpeg setting LD_LIBRARY_PATH={hip_lib_path}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{hip_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(hip_lib_path)
    else:
        logging.info("++ rocjpeg tests only supported on Linux")
        sys.exit(0)


def execute_tests(env):
    rocjpeg_test_dir = THEROCK_TEST_DIR / "rocjpeg-test"
    rocjpeg_test_dir.mkdir(parents=True, exist_ok=True)

    # rocjpeg tests are shipped as CMake source and must be built on the target
    # machine. This serves two purposes:
    # 1. Verifies that the installed rocjpeg headers and libraries are functional.
    # 2. Some test dependencies are not bundled in the TheRock artifacts and must
    #    be linked from the system at build time.
    cmd = [
        "cmake",
        "-GNinja",
        ROCJPEG_TEST_PATH,
    ]
    logging.info(f"++ Exec [{rocjpeg_test_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocjpeg_test_dir, check=True, env=env)

    cmd = [
        "ctest",
        "-N",
    ]
    logging.info(f"++ Exec [{rocjpeg_test_dir}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=rocjpeg_test_dir,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    logging.info(ctest_list.stdout)
    match = re.search(r"Total Tests:\s*(\d+)", ctest_list.stdout)
    if match is None:
        raise RuntimeError(
            "Failed to determine CTest test count from `ctest -N` output"
        )
    if int(match.group(1)) == 0:
        raise RuntimeError("CTest discovered zero rocjpeg tests")

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ]
    logging.info(f"++ Exec [{rocjpeg_test_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=rocjpeg_test_dir, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
