#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and run the rocdecode project's installed test suite.

This runner is installed with the rocdecode test source payload. It configures
and builds those installed tests on the target machine, then executes the
generated CTest suite.
"""

import argparse
import logging
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)

PROJECT_NAME = "rocdecode"

HELP_EPILOG = f"""\
This script runs the {PROJECT_NAME} project test suite from an installed ROCm
payload. When run from an installed location, it discovers:

  <rocm-prefix>/share/{PROJECT_NAME}/test

The installed tests are CMake source files, so the runner configures and builds
them before invoking CTest.

Environment variables:
  ROCM_PATH          ROCm install prefix override.
  TEST_BUILD_DIR     Build directory for the installed test source.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build and run the {PROJECT_NAME} project test suite.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "test" and script_dir.parent.name == PROJECT_NAME:
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed rocdecode test layout. "
        "Set ROCM_PATH explicitly."
    )


def get_rocm_lib_dir(rocm_path: Path) -> Path:
    for name in ("lib", "lib64"):
        candidate = rocm_path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find ROCm library directory under {rocm_path}")


def setup_env(env: Dict[str, str], rocm_path: Path) -> bool:
    env["ROCM_PATH"] = str(rocm_path)
    logging.info(f"++ rocdecode setting ROCM_PATH={rocm_path}")
    if platform.system() == "Linux":
        hip_lib_path = get_rocm_lib_dir(rocm_path)
        logging.info(f"++ rocdecode setting LD_LIBRARY_PATH={hip_lib_path}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{hip_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(hip_lib_path)
        return True
    else:
        logging.info("++ rocdecode tests only supported on Linux")
        return False


def execute_tests(env: Dict[str, str], test_source_dir: Path, build_dir: Path) -> None:
    if not test_source_dir.is_dir():
        raise FileNotFoundError(f"rocdecode tests not found in {test_source_dir}")

    build_dir.mkdir(parents=True, exist_ok=True)

    # rocdecode tests are shipped as CMake source and must be built on the target
    # machine. This serves two purposes:
    # 1. Verifies that the installed rocdecode headers and libraries are functional.
    # 2. Some test dependencies (e.g. video codec libraries) are not bundled in the
    #    installed test payload and must be linked from the system at build time.
    cmd = [
        "cmake",
        "-GNinja",
        str(test_source_dir),
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=build_dir, check=True, env=env)

    cmd = [
        "ctest",
        "-N",
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=build_dir,
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
        raise RuntimeError("CTest discovered zero rocdecode tests")

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ]
    logging.info(f"++ Exec [{build_dir}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=build_dir, check=True, env=env)


def main() -> None:
    parse_args()
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    test_source_dir = rocm_path / "share" / PROJECT_NAME / "test"
    build_dir = Path(os.getenv("TEST_BUILD_DIR") or Path.cwd() / "rocdecode-test")
    build_dir = build_dir.resolve()

    env = os.environ.copy()
    if not setup_env(env, rocm_path):
        return
    execute_tests(env, test_source_dir, build_dir)


if __name__ == "__main__":
    main()
