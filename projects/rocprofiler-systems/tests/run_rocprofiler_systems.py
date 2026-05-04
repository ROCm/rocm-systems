#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

TEST_FILTER = (
    "not TestOpenMPTarget and not (TestTranspose and runtime_instrument) "
    "and not TestGPUConnect"
)


def format_command(cmd) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def derive_rocm_path(script_dir: Path) -> Path:
    if script_dir.name == "tests" and script_dir.parent.name == "rocprofiler-systems":
        if script_dir.parent.parent.name == "share":
            return script_dir.parent.parent.parent
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed rocprofiler-systems test "
        "layout. Set ROCM_PATH explicitly."
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    )
    rocm_bin_dir = Path(os.getenv("ROCM_BIN_DIR") or rocm_path / "bin").resolve()
    tests_dir = rocm_path / "share" / "rocprofiler-systems" / "tests"
    if not tests_dir.is_dir():
        raise FileNotFoundError(f"Could not find rocprofiler-systems tests: {tests_dir}")

    env = os.environ.copy()
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{rocm_bin_dir}{os.pathsep}{existing_path}"
        if existing_path
        else str(rocm_bin_dir)
    )
    env["ROCM_PATH"] = str(rocm_path)
    env["ROCPROFSYS_INSTALL_DIR"] = str(rocm_path)

    examples_lib_dir = rocm_path / "share" / "rocprofiler-systems" / "examples" / "lib"
    existing_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{examples_lib_dir}{os.pathsep}{existing_ld_path}"
        if existing_ld_path
        else str(examples_lib_dir)
    )

    pytest_package_exec = tests_dir / "rocprofsys-tests.pyz"
    if not pytest_package_exec.is_file():
        raise FileNotFoundError(f"Could not find test package: {pytest_package_exec}")
    cmd = [
        sys.executable,
        str(pytest_package_exec),
        "-k",
        TEST_FILTER,
        "--junit-xml=junit.xml",
        "--ci-mode",
        "--log-cli-level=info",
    ]

    logging.info(f"++ Exec [{rocm_path}]$ {format_command(cmd)}")
    subprocess.run(cmd, cwd=rocm_path, check=True, env=env)


if __name__ == "__main__":
    main()
