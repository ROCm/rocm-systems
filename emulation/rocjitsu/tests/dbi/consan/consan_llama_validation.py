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
    "rms-norm": {"tolerance": 1.0e-5, "n_embd": 128, "n_tokens": 1},
    # The corpus's 128-element setup smoke is too small for a dropped
    # synchronization pair to have a reproducible semantic effect. A
    # 1024-element embedding remains compact while exercising enough native
    # matvec workgroups for the independent CPU oracle to expose the fault.
    "mul-mat-vec-q": {"tolerance": 2.0e-2, "n_embd": 1024, "n_tokens": 1},
}


def _write_oracle_result(outcome: str, detail: object) -> None:
    result_path = os.environ.get("CONSAN_ROW_RESULT_PATH") or os.environ.get(
        "CONSAN_WORKLOAD_RESULT_PATH"
    )
    if not result_path:
        return
    payload = {
        "schema_version": 1,
        "oracle": outcome,
        "detail": detail,
        "source_diagnostics": {
            "outcome": "not_applicable",
            "count": None,
            "expectation": "not_applicable",
            "detail": "llama.cpp numeric oracle has no separate source-diagnostic channel",
        },
    }
    path = Path(result_path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    workload = WORKLOADS[args.workload]
    common = [
        str(args.executable),
        "--n-embd",
        str(workload["n_embd"]),
        "--n-tokens",
        str(workload["n_tokens"]),
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
                oracle_passed = max_abs_error <= workload["tolerance"]
                detail = "pass" if oracle_passed else "tolerance exceeded"
        except (OSError, ValueError) as error:
            detail = str(error)

    result = {
        f"llama-{args.workload}": {
            "median_ms": gpu_elapsed_ms,
            "oracle": "independent-cpu-binary-output",
            "oracle_passed": oracle_passed,
            "max_abs_error": max_abs_error,
            "tolerance": workload["tolerance"],
            "gpu_returncode": gpu_returncode,
            "cpu_returncode": cpu_returncode,
            "detail": detail,
        }
    }
    _write_oracle_result("pass" if oracle_passed else "fail", result)
    print(json.dumps(result, sort_keys=True))
    if gpu_returncode != 0:
        return gpu_returncode
    if cpu_returncode != 0:
        return cpu_returncode
    return 0 if oracle_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
