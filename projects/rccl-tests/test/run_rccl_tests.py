# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path

PROJECT_NAME = "rccl-tests"

TEST_EXECUTABLES = [
    "all_gather_perf",
    "alltoallv_perf",
    "broadcast_perf",
    "alltoall_perf",
    "all_reduce_perf",
    "reduce_perf",
    "hypercube_perf",
    "gather_perf",
    "scatter_perf",
    "sendrecv_perf",
    "reduce_scatter_perf",
]

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run installed rccl-tests binaries.")
    parser.add_argument(
        "--rocm-path",
        type=Path,
        default=os.getenv("ROCM_PATH"),
        help="ROCm install prefix. Defaults to ROCM_PATH or the runner location.",
    )
    parser.add_argument(
        "--bin-dir",
        type=Path,
        help="Directory containing rccl-tests binaries. Defaults to <rocm-path>/bin.",
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

    # Installed at <prefix>/share/rccl-tests/tests.
    if (
        script_path.parent.name == "tests"
        and script_path.parents[1].name == PROJECT_NAME
    ):
        return script_path.parents[3].resolve()

    raise SystemExit("ROCM_PATH is required when the runner is not installed.")


def resolve_executable(bin_dir: Path, name: str) -> Path:
    suffixes = [".exe", ""] if os.name == "nt" else ["", ".exe"]
    for suffix in suffixes:
        candidate = bin_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit(f"{name} was not found in {bin_dir}")


def run_rccl_tests(bin_dir: Path, work_dir: Path) -> None:
    for executable in TEST_EXECUTABLES:
        cmd = [str(resolve_executable(bin_dir, executable))]
        logging.info("++ Exec [%s]$ %s", work_dir, shlex.join(cmd))
        subprocess.run(cmd, cwd=work_dir, check=True)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    rocm_path = get_rocm_path(script_path, args.rocm_path)
    bin_dir = args.bin_dir.resolve() if args.bin_dir is not None else rocm_path / "bin"
    work_dir = (
        args.work_dir.resolve() if args.work_dir is not None else rocm_path
    )

    run_rccl_tests(bin_dir, work_dir)


if __name__ == "__main__":
    main()
