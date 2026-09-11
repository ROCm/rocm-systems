#!/usr/bin/env python3
"""Run one verified hipBLASLt/Tensile GEMM and emit benchmark measurements."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="
INSTRUMENTATION_RE = re.compile(
    r"^(?:\[rocjitsu-dbi-hooks\] )?ConSan instrumentation timing " r"total_ns=(\d+)$",
    re.MULTILINE,
)
KERNEL_RE = re.compile(r"^\s*--kernel name:\s*(.+?)\s*$", re.MULTILINE)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aorta-dir", type=Path, required=True)
    parser.add_argument("--hipblaslt-bench", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args(argv)


def _performance_rows(output: str) -> list[dict[str, str]]:
    """Parse hipblaslt-bench's adjacent CSV header/value records."""
    lines = output.splitlines()
    rows: list[dict[str, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("[") or ",us" not in stripped:
            continue
        colon = stripped.find(":")
        if colon < 0:
            continue
        header = next(csv.reader(io.StringIO(stripped[colon + 1 :])))
        value_index = index + 1
        while value_index < len(lines) and not lines[value_index].strip():
            value_index += 1
        if value_index == len(lines):
            continue
        values = next(csv.reader(io.StringIO(lines[value_index].strip())))
        if len(header) == len(values):
            rows.append(dict(zip(header, values, strict=True)))
    return rows


def _positive_floats(rows: list[dict[str, str]], field: str) -> tuple[float, ...]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if len(values) != 2:
        raise RuntimeError(
            f"expected two positive hipblaslt-bench {field!r} values, found {values}"
        )
    return tuple(values)


def _nonnegative_floats(rows: list[dict[str, str]], field: str) -> tuple[float, ...]:
    try:
        values = tuple(float(row[field]) for row in rows)
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"invalid hipblaslt-bench {field!r} value") from error
    if len(values) != 2 or any(value < 0 for value in values):
        raise RuntimeError(
            f"expected two nonnegative hipblaslt-bench {field!r} values, found {values}"
        )
    return values


def _instrumentation_ms(output: str) -> float:
    matches = INSTRUMENTATION_RE.findall(output)
    if "RJ_CONSAN_MODE" not in os.environ:
        if matches:
            raise RuntimeError(
                "native hipBLASLt run unexpectedly reported ConSan timing"
            )
        return 0.0
    if len(matches) != 1:
        raise RuntimeError(
            "instrumented hipBLASLt run did not report exactly one ConSan timing record"
        )
    return int(matches[0]) / 1_000_000.0


def _yaml_text(common: Path, m: int, n: int, k: int, iterations: int) -> str:
    test = f"""\
  function: matmul
  a_type: f16_r
  b_type: f16_r
  c_type: f16_r
  d_type: f16_r
  compute_type: c_f32_r
  scale_type: f32_r
  M: {m}
  N: {n}
  K: {k}
  transA: N
  transB: N
  alpha: 1.0
  beta: 0.0
  iters: {iterations}
  cold_iters: 0
  norm_check: 1
  allclose_check: 1
  norm_check_assert: true
  use_gpu_timer: true
  algo_method: 0
  requested_solution_num: 1
  print_kernel_info: true
"""
    return (
        "---\n"
        f"include: {common}\n\n"
        "Tests:\n"
        "- name: consan-benchmark-run1\n"
        f"{test}"
        "- name: consan-benchmark-run2\n"
        f"{test}"
        "...\n"
    )


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not (args.aorta_dir.resolve() / "src" / "aorta").is_dir():
        raise SystemExit(f"Aorta checkout not found under {args.aorta_dir.resolve()}")
    if not args.hipblaslt_bench.is_file():
        raise SystemExit(f"hipblaslt-bench not found: {args.hipblaslt_bench}")
    try:
        config = json.loads(args.config_json)
        m = int(config["m"])
        n = int(config["n"])
        k = int(config["k"])
        iterations = int(config.get("iterations", 10))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid hipBLASLt benchmark config: {error}") from error
    if min(m, n, k, iterations) < 1:
        raise SystemExit("hipBLASLt dimensions and iteration count must be positive")

    common = args.hipblaslt_bench.resolve().parent / "hipblaslt_common.yaml"
    if not common.is_file():
        raise SystemExit(f"hipblaslt_common.yaml not found beside benchmark: {common}")
    data_path = Path(tempfile.gettempdir()) / (
        f"rocjitsu-consan-hipblaslt-{os.getpid()}-{time.time_ns()}.yaml"
    )
    data_path.write_text(_yaml_text(common, m, n, k, iterations), encoding="utf-8")
    command = [str(args.hipblaslt_bench.resolve()), "--yaml", str(data_path)]
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - start) * 1000.0
    output = process.stdout + process.stderr
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if process.returncode != 0:
        raise SystemExit(process.returncode)

    rows = _performance_rows(output)
    selected_kernels = tuple(KERNEL_RE.findall(output))
    kernel_names = tuple(dict.fromkeys(selected_kernels))
    if len(selected_kernels) != 2 or len(kernel_names) != 1:
        raise RuntimeError(
            "expected both hipBLASLt runs to select the same single kernel; "
            f"found {selected_kernels}"
        )
    runtime_ms = tuple(value / 1000.0 for value in _positive_floats(rows, "us"))
    norm_errors = _nonnegative_floats(rows, "norm_error")
    instrumentation_ms = _instrumentation_ms(output)
    runs = [
        {
            "index": index + 1,
            "result": {
                "passed": True,
                "metrics": {
                    "latency_ms": runtime_ms[index],
                    "parameter_count": m * k + k * n,
                    "matrix_m": m,
                    "matrix_n": n,
                    "matrix_k": k,
                    "iterations": iterations,
                    "norm_error": norm_errors[index],
                },
            },
            "runtime_ms": runtime_ms[index],
            "instrumentation_ms": 0.0,
        }
        for index in range(2)
    ]
    payload = {
        "result": runs[0]["result"],
        "runs": runs,
        "phase_ms": {"setup": wall_ms, "run": runtime_ms[0]},
        "instrumentation_ms": {
            "before_run": instrumentation_ms,
            "during_run": 0.0,
            "total": instrumentation_ms,
        },
        "peak_device_memory": {},
        "runtime": {
            "hipblaslt_bench": str(args.hipblaslt_bench.resolve()),
            "data_file": str(data_path),
            "target": os.environ.get("HIP_TARGET", "unknown"),
        },
        "kernel_names": list(kernel_names),
        "kernel_stats": [
            {
                "name": kernel_names[0],
                "dispatches": 2 * iterations,
                "device_time_us": sum(runtime_ms) * 1000.0 * iterations,
            }
        ],
    }
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
