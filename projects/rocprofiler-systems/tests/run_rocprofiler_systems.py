#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the rocprofiler-systems project's installed test suite via CTest.

This runner is installed with the rocprofiler-systems test payload and
executes the CTest definitions shipped by that project
(``CTestTestfile.cmake``), which in turn invoke the standalone
``rocprofsys-tests.pyz`` package.

This script is intended primarily as a CI / automated entry point. The
hardcoded ``EXCLUDED_TESTS`` / ``EXCLUDED_LABELS`` / ``QUICK_TESTS_REGEX``
lists below curate which tests run in install mode. Developers running
the suite locally should prefer ``ctest --test-dir <prefix>/share/...``
to bypass these filters.

Environment variables:
  ROCPROFSYS_INSTALL_DIR  rocprofiler-systems install prefix override.
  ROCM_PATH               ROCm dependency prefix. Defaults to /opt/rocm.
  ROCM_BIN_DIR            Directory containing ROCm command-line tools.
  TEST_TYPE               'full' (default) or 'quick'. 'quick' restricts
                          the run to a curated short subset (~15 min).
"""

import argparse
import getpass
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PROJECT_NAME = "rocprofiler-systems"

# CTest test-name regexes (CTest --exclude-regex) for tests that should not
# run in installed mode. Joined with '|' to form a single regex alternation.
# Always excluded until the relevant issue is fixed (AIPROFSYST-441).
EXCLUDED_TESTS = [
    "transferbench-sys-run",
    "fork.*",
    "openmp-target.*",
    "roctx-sampling",
    "roctx-runtime-instrument",
    "jacobi-usm-sys-run",
    "jacobi-roctx.*",
    "jpeg-decode.*",
    "matrix-exponential.*",
    "scratch-memory.*",
    "selective-region-region-1-filter.*",
    "selective-region-region-2-and-3.*",
    "selective-region-no-marker-region-1-filter.*",
    "shmem-pingpong.*",
    "video-decode.*",
]

# CTest labels (CTest --label-exclude) for tests that should not run in
# installed mode. Each pytest marker becomes a CTest label in the generated
# CTestTestfile.cmake. Excluded by default (AIPROFSYST-441).
EXCLUDED_LABELS = [
    "annotate",
    "mpi",
    "julia",
    "attach",
    "lulesh",
    "network",
    "overflow",
    "thread_limit",
    "rockoff",
]

# Curated short subset for TEST_TYPE=quick (target ~15 minutes).
QUICK_TESTS_REGEX = [
    "transpose.*",
    "rocprofiler-systems.*",  # Binary tests
    "config.*",
    "openmp.*",
    "roctx.*",
    "trace-time-window.*",
]

HELP_EPILOG = f"""\
This script runs the {PROJECT_NAME} project test suite from an installed
payload via CTest. When run from an installed location, it discovers:

  <install-prefix>/share/{PROJECT_NAME}/tests/CTestTestfile.cmake

Environment variables:
  ROCPROFSYS_INSTALL_DIR  rocprofiler-systems install prefix override.
  ROCM_PATH               ROCm dependency prefix. Defaults to /opt/rocm.
  ROCM_BIN_DIR            Directory containing ROCm command-line tools.
  TEST_TYPE               'full' (default) or 'quick' for a curated subset.
"""


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {PROJECT_NAME} project test suite via CTest.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def derive_install_dir(script_dir: Path) -> Path:
    if script_dir.name == "tests" and script_dir.parent.name == PROJECT_NAME:
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCPROFSYS_INSTALL_DIR from an installed "
        "rocprofiler-systems test layout. Set ROCPROFSYS_INSTALL_DIR explicitly."
    )


def default_output_dir() -> Path:
    try:
        user = getpass.getuser()
    except Exception:
        user = str(os.getuid()) if hasattr(os, "getuid") else "unknown"
    return Path(tempfile.gettempdir()) / user / "rocprof-sys-pytest-output"


def prepend_path(env, name: str, path: Path) -> None:
    existing_value = env.get(name, "")
    path_value = str(path)
    env[name] = (
        f"{path_value}{os.pathsep}{existing_value}" if existing_value else path_value
    )


def main() -> None:
    parse_args()
    script_dir = Path(__file__).resolve().parent
    install_dir_env = os.getenv("ROCPROFSYS_INSTALL_DIR")
    install_dir = (
        Path(install_dir_env).resolve()
        if install_dir_env
        else derive_install_dir(script_dir)
    )
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env or "/opt/rocm").resolve()
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()
    tests_dir = install_dir / "share" / PROJECT_NAME / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find rocprofiler-systems tests: {tests_dir}"
        )
    ctest_file = tests_dir / "CTestTestfile.cmake"
    if not ctest_file.is_file():
        raise FileNotFoundError(f"Could not find CTest definitions: {ctest_file}")

    test_output_dir = default_output_dir()
    test_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    prepend_path(env, "PATH", rocm_bin_dir)
    prepend_path(env, "PATH", install_dir / "bin")
    env["ROCM_PATH"] = str(rocm_path)
    env["ROCPROFSYS_INSTALL_DIR"] = str(install_dir)

    examples_lib_dir = install_dir / "share" / PROJECT_NAME / "examples" / "lib"
    prepend_path(env, "LD_LIBRARY_PATH", examples_lib_dir)

    test_type = os.getenv("TEST_TYPE", "full").lower()

    ctest_base = ["ctest", "--test-dir", str(tests_dir)]

    cmd = ctest_base + [
        "--output-on-failure",
        "--exclude-regex",
        f"{'|'.join(EXCLUDED_TESTS)}",
        "--label-exclude",
        f"{'|'.join(EXCLUDED_LABELS)}",
        "--repeat",
        "until-pass:3",
    ]
    if test_type == "quick":
        cmd.extend(["--tests-regex", "|".join(QUICK_TESTS_REGEX)])

    logging.info(f"++ Exec [{test_output_dir}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=test_output_dir, check=True, env=env)


if __name__ == "__main__":
    main()
