#!/usr/bin/env python3
"""Runs one llama.cpp GPU case against its independent CPU oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


WORKLOADS = {
    "rms-norm": {"tolerance": 1.0e-5},
    "mul-mat-vec-q": {"tolerance": 1.0e-2},
}


def _read_f32(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} has a non-F32 byte size: {len(data)}")
    return tuple(value[0] for value in struct.iter_unpack("=f", data))


def _cpu_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("RJ_CONSAN_")
        and name
        not in {
            "HIP_TARGET",
            "HSA_TOOLS_LIB",
            "HSA_TOOLS_ROCPROFILER_V1_TOOLS",
        }
    }


def _run(
    command: list[str], environment: dict[str, str] | None = None
) -> tuple[int, float]:
    start = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed_ms = (time.monotonic() - start) * 1.0e3
    if completed.stdout:
        print(completed.stdout, file=sys.stderr, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.returncode, elapsed_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_output = args.output_dir / "gpu-output.bin"
    cpu_output = args.output_dir / "cpu-output.bin"
    common = [
        str(args.executable),
        "--n-embd",
        "128",
        "--n-tokens",
        "1",
    ]
    gpu_returncode, gpu_elapsed_ms = _run([*common, "--output", str(gpu_output)])
    cpu_returncode, _ = _run(
        [*common, "--validate", "--output", str(cpu_output)],
        _cpu_environment(),
    )

    oracle_passed = False
    max_abs_error: float | None = None
    detail = "child process failed"
    if gpu_output.is_file() and cpu_output.is_file():
        try:
            gpu = _read_f32(gpu_output)
            cpu = _read_f32(cpu_output)
            if len(gpu) != len(cpu):
                detail = f"element-count mismatch: GPU={len(gpu)} CPU={len(cpu)}"
            elif not gpu:
                detail = "empty output"
            elif not all(math.isfinite(value) for value in (*gpu, *cpu)):
                detail = "non-finite output"
            else:
                max_abs_error = max(abs(left - right) for left, right in zip(gpu, cpu))
                oracle_passed = max_abs_error <= WORKLOADS[args.workload]["tolerance"]
                detail = "pass" if oracle_passed else "tolerance exceeded"
        except (OSError, ValueError) as error:
            detail = str(error)

    print(
        json.dumps(
            {
                f"llama-{args.workload}": {
                    "median_ms": gpu_elapsed_ms,
                    "oracle": "independent-cpu-binary-output",
                    "oracle_passed": oracle_passed,
                    "max_abs_error": max_abs_error,
                    "tolerance": WORKLOADS[args.workload]["tolerance"],
                    "gpu_returncode": gpu_returncode,
                    "cpu_returncode": cpu_returncode,
                    "detail": detail,
                }
            },
            sort_keys=True,
        )
    )
    if gpu_returncode != 0:
        return gpu_returncode
    if cpu_returncode != 0:
        return cpu_returncode
    return 0 if oracle_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
