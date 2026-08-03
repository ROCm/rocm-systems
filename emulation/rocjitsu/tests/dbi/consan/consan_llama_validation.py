#!/usr/bin/env python3
"""Runs one llama.cpp GPU case against its independent CPU oracle."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
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
) -> tuple[int, float, str]:
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
    return completed.returncode, elapsed_ms, completed.stdout or ""


def _gpu_timing(output: str, expected_iterations: int) -> tuple[float, float]:
    number = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    matches = re.findall(
        r"^llama_gpu_timing timer=hip-event "
        rf"aggregate_ms=({number}) "
        r"iterations=([0-9]+) "
        rf"per_iteration_ms=({number})$",
        output,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(f"expected one llama GPU timing row, found {len(matches)}")
    aggregate_ms = float(matches[0][0])
    iterations = int(matches[0][1])
    per_iteration_ms = float(matches[0][2])
    if iterations != expected_iterations:
        raise ValueError(
            f"llama GPU timing iteration mismatch: {iterations} != {expected_iterations}"
        )
    if (
        not math.isfinite(aggregate_ms)
        or not math.isfinite(per_iteration_ms)
        or aggregate_ms <= 0.0
        or per_iteration_ms <= 0.0
        or not math.isclose(aggregate_ms / iterations, per_iteration_ms, rel_tol=2.0e-6)
    ):
        raise ValueError("llama GPU timing row is inconsistent")
    return aggregate_ms, per_iteration_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--benchmark-iterations", type=int)
    parser.add_argument("--benchmark-warmup-iterations", type=int, default=5)
    parser.add_argument("--minimum-timed-ms", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.n_embd is not None and args.n_embd <= 0:
        parser.error("--n-embd must be positive")
    if args.benchmark_iterations is not None and args.benchmark_iterations <= 0:
        parser.error("--benchmark-iterations must be positive")
    if args.benchmark_warmup_iterations <= 0:
        parser.error("--benchmark-warmup-iterations must be positive")
    if not math.isfinite(args.minimum_timed_ms) or args.minimum_timed_ms < 0.0:
        parser.error("--minimum-timed-ms must be finite and nonnegative")
    if args.minimum_timed_ms > 0.0 and args.benchmark_iterations is None:
        parser.error("--minimum-timed-ms requires --benchmark-iterations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_output = args.output_dir / "gpu-output.bin"
    cpu_output = args.output_dir / "cpu-output.bin"
    workload = dict(WORKLOADS[args.workload])
    if args.n_embd is not None:
        workload["n_embd"] = args.n_embd
    common = [
        str(args.executable),
        "--n-embd",
        str(workload["n_embd"]),
        "--n-tokens",
        str(workload["n_tokens"]),
    ]
    gpu_command = [*common, "--output", str(gpu_output)]
    if args.benchmark_iterations is not None:
        gpu_command.extend(
            [
                "--benchmark-iterations",
                str(args.benchmark_iterations),
                "--benchmark-warmup-iterations",
                str(args.benchmark_warmup_iterations),
            ]
        )
    gpu_returncode, gpu_elapsed_ms, gpu_stdout = _run(gpu_command)
    cpu_returncode, _, _ = _run(
        [*common, "--validate", "--output", str(cpu_output)],
        _cpu_environment(),
    )

    device_aggregate_ms: float | None = None
    device_ms: float | None = None
    timing_detail = "not requested"
    if args.benchmark_iterations is not None:
        try:
            device_aggregate_ms, device_ms = _gpu_timing(
                gpu_stdout, args.benchmark_iterations
            )
            timing_detail = (
                "pass"
                if device_aggregate_ms >= args.minimum_timed_ms
                else (
                    "timed aggregate is below minimum: "
                    f"{device_aggregate_ms} < {args.minimum_timed_ms}"
                )
            )
            if timing_detail != "pass":
                device_aggregate_ms = None
                device_ms = None
        except ValueError as error:
            timing_detail = str(error)

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
            "device_median_ms": device_ms,
            "device_samples_ms": [device_ms] if device_ms is not None else None,
            "device_timer": "hip-event" if device_ms is not None else None,
            "device_timed_aggregate_ms": device_aggregate_ms,
            "device_timed_iterations": args.benchmark_iterations,
            "device_timing_detail": timing_detail,
            "timed_aggregate_minimum_ms": args.minimum_timed_ms,
            "benchmark_iterations": args.benchmark_iterations,
            "timed_aggregate_ms": device_aggregate_ms,
            "timing_source": "hip-event" if device_ms is not None else None,
            "benchmark_shape": [workload["n_embd"], workload["n_tokens"]],
        }
    }
    _write_oracle_result("pass" if oracle_passed else "fail", result)
    print(json.dumps(result, sort_keys=True))
    if gpu_returncode != 0:
        return gpu_returncode
    if cpu_returncode != 0:
        return cpu_returncode
    if args.benchmark_iterations is not None and device_ms is None:
        return 1
    return 0 if oracle_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
