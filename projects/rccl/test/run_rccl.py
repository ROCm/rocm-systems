#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the RCCL project's installed unit test suite.

This runner is installed with RCCL and executes the rccl-UnitTests binary
shipped by that project when at least two GPUs are visible.
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional

PROJECT_NAME = "rccl"

logging.basicConfig(level=logging.INFO)

HELP_EPILOG = f"""\
This script runs the {PROJECT_NAME} project unit test suite from an installed
ROCm payload. When run from an installed location, it discovers:

  <rocm-prefix>/bin/rccl-UnitTests[.exe]

The test suite requires at least two visible GPUs.
"""


def path_from_env(name: str) -> Optional[Path]:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the {PROJECT_NAME} project unit test suite.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=path_from_env("ROCM_PATH"),
        help=(
            "ROCm install prefix used by the RCCL tests. Defaults to ROCM_PATH "
            "or the installed runner location."
        ),
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        help="Directory containing rccl-UnitTests. Defaults to <rocm-path>/bin.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory for test execution. Defaults to <rocm-path>.",
    )
    return parser.parse_args()


def get_rocm_path(script_path: Path, rocm_path: Optional[Path]) -> Path:
    if rocm_path is not None:
        return rocm_path.resolve()

    # Installed at <prefix>/share/rccl/tests.
    if (
        script_path.parent.name == "tests"
        and script_path.parents[1].name == PROJECT_NAME
        and script_path.parents[2].name == "share"
    ):
        return script_path.parents[3].resolve()

    raise RuntimeError("ROCM_PATH is required when the runner is not installed.")


def resolve_executable(bin_dir: Path, name: str) -> Path:
    suffixes = [".exe", ""] if os.name == "nt" else ["", ".exe"]
    for suffix in suffixes:
        candidate = bin_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{name} was not found in {bin_dir}")


def get_visible_gpu_count(env: Dict[str, str], bin_dir: Path) -> int:
    rocminfo_path = bin_dir / ("rocminfo.exe" if os.name == "nt" else "rocminfo")
    rocminfo = str(rocminfo_path) if rocminfo_path.exists() else "rocminfo"
    result = subprocess.run(
        [rocminfo],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )

    pattern = re.compile(r"^\s*Name:\s+gfx[0-9a-z]+$", re.IGNORECASE)
    return sum(1 for line in result.stdout.splitlines() if pattern.match(line))


def run_rccl_unit_tests(bin_dir: Path, work_dir: Path) -> None:
    env = os.environ.copy()
    gpu_count = get_visible_gpu_count(env, bin_dir)
    logging.info("Visible GPU count: %s", gpu_count)

    if gpu_count < 2:
        logging.info("Skipping RCCL unit tests: <2 GPUs visible")
        return

    env.setdefault("HIP_VISIBLE_DEVICES", "0,1")
    env["UT_MIN_GPUS"] = "2"
    env["UT_MAX_GPUS"] = "2"
    env["UT_POW2_GPUS"] = "1"
    env["UT_PROCESS_MASK"] = "1"

    cmd = [str(resolve_executable(bin_dir, "rccl-UnitTests"))]
    logging.info("++ Exec [%s]$ %s", work_dir, shlex.join(cmd))
    subprocess.run(cmd, cwd=work_dir, check=True, env=env)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    rocm_path = get_rocm_path(script_path, args.rocm_path)
    bin_dir = args.bin_dir.resolve() if args.bin_dir is not None else rocm_path / "bin"
    work_dir = (
        args.work_dir.resolve() if args.work_dir is not None else rocm_path
    )

    run_rccl_unit_tests(bin_dir, work_dir)


if __name__ == "__main__":
    main()
