#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the rocprofiler-systems project's installed pytest suite.

This runner is installed with the rocprofiler-systems test payload and executes
the standalone rocprofsys-tests.pyz package shipped by that project.
"""

import argparse
import getpass
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)

PROJECT_NAME = "rocprofiler-systems"
TEST_FILTER = (
    "not TestOpenMPTarget and not (TestTranspose and runtime_instrument) "
    "and not TestGPUConnect"
)

HELP_EPILOG = f"""\
This script runs the {PROJECT_NAME} project test suite from an installed
payload. When run from an installed location, it discovers:

  <install-prefix>/share/{PROJECT_NAME}/tests/rocprofsys-tests.pyz

Environment variables:
  ROCPROFSYS_INSTALL_DIR  rocprofiler-systems install prefix override.
  ROCM_PATH               ROCm dependency prefix. Defaults to /opt/rocm.
  ROCM_BIN_DIR            Directory containing ROCm command-line tools.
"""


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {PROJECT_NAME} project pytest suite.",
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
        raise FileNotFoundError(f"Could not find rocprofiler-systems tests: {tests_dir}")
    test_output_dir = default_output_dir()
    test_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    prepend_path(env, "PATH", rocm_bin_dir)
    prepend_path(env, "PATH", install_dir / "bin")
    env["ROCM_PATH"] = str(rocm_path)
    env["ROCPROFSYS_INSTALL_DIR"] = str(install_dir)

    examples_lib_dir = install_dir / "share" / PROJECT_NAME / "examples" / "lib"
    prepend_path(env, "LD_LIBRARY_PATH", examples_lib_dir)

    pytest_package_exec = tests_dir / "rocprofsys-tests.pyz"
    if not pytest_package_exec.is_file():
        raise FileNotFoundError(f"Could not find test package: {pytest_package_exec}")
    cmd = [
        sys.executable,
        str(pytest_package_exec),
        "-k",
        TEST_FILTER,
        f"--junit-xml={test_output_dir / 'junit.xml'}",
        "--ci-mode",
        "--log-cli-level=info",
    ]

    logging.info(f"++ Exec [{test_output_dir}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=test_output_dir, check=True, env=env)


if __name__ == "__main__":
    main()
