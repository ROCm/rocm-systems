#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

PROJECT_NAME = "rccl"

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run installed RCCL unit tests.")
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=os.getenv("ROCM_PATH"),
        help="ROCm install prefix. Defaults to ROCM_PATH or the runner location.",
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


def get_rocm_path(script_path: Path, rocm_path: Path | None) -> Path:
    if rocm_path is not None:
        return rocm_path.resolve()

    # Installed at <prefix>/share/rccl/tests.
    if (
        script_path.parent.name == "tests"
        and script_path.parents[1].name == PROJECT_NAME
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


def get_visible_gpu_count(env: dict[str, str], bin_dir: Path) -> int:
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
