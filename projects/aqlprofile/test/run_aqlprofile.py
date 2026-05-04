#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the aqlprofile project's installed test suite.

This runner is installed with the hsa-amd-aqlprofile test payload and executes
the run_tests.sh script shipped by that project. It is intended to be called
directly from CI or by developers with an installed ROCm tree.
"""

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

HELP_EPILOG = """\
This script runs the aqlprofile project test suite from an installed ROCm
payload. When run from an installed location, it discovers:

  <rocm-prefix>/share/hsa-amd-aqlprofile/run_tests.sh

Environment variables:
  ROCM_PATH          ROCm install prefix override.
  LD_LIBRARY_PATH    Existing library path to append after the ROCm lib dir.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the aqlprofile project test suite.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        test_script = candidate / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"
        if test_script.is_file():
            return candidate
    raise RuntimeError(
        "Could not derive ROCM_PATH from an installed hsa-amd-aqlprofile test "
        "layout. Set ROCM_PATH explicitly."
    )


def get_rocm_lib_dir(rocm_path: Path) -> Path:
    for name in ("lib", "lib64"):
        candidate = rocm_path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find ROCm library directory under {rocm_path}")


def main() -> None:
    parse_args()
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    test_script = rocm_path / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"
    if not test_script.is_file():
        raise FileNotFoundError(f"Could not find test script: {test_script}")

    env = os.environ.copy()
    lib_path = get_rocm_lib_dir(rocm_path)
    old_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{lib_path}{os.pathsep}{old_ld_library_path}"
        if old_ld_library_path
        else str(lib_path)
    )

    cmd = [str(test_script)]
    logging.info(f"++ Exec [{rocm_path}]$ {shlex.join(cmd)}")

    subprocess.run(
        cmd,
        cwd=rocm_path,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
