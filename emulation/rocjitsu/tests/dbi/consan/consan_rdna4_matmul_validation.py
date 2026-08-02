#!/usr/bin/env python3
"""Runs one production RDNA4 matmul through its checked native harness."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

WORKLOADS = {
    "fp16-production": "fp16_wmma_tiled_prod_16x8_noop",
    "fp8-production": "fp8_wmma_tiled_prod_16x8_noop",
}
MINIMUM_TIMED_MS = 250.0
CHECK_SHAPE = (256, 128, 128)
BENCHMARK_SHAPE = (4096, 4096, 4096)


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
            "detail": "the native numeric oracle has no separate diagnostic channel",
        },
    }
    path = Path(result_path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment(variant: str, repetitions: int, phase: str) -> dict[str, str]:
    clean = phase in {"clean", "cold"}
    environment = dict(os.environ)
    environment.update(
        {
            "FILTER": variant,
            "FILTER_EXACT": "1",
            "CHECK_M": str(CHECK_SHAPE[0]),
            "CHECK_N": str(CHECK_SHAPE[1]),
            "CHECK_K": str(CHECK_SHAPE[2]),
            "BENCH_M": str(BENCHMARK_SHAPE[0]),
            "BENCH_N": str(BENCHMARK_SHAPE[1]),
            "BENCH_K": str(BENCHMARK_SHAPE[2]),
            "BENCH_REPEATS": str(repetitions),
            "BENCH_CALIBRATION_ITERS": "10",
            "MIN_MS": str(MINIMUM_TIMED_MS),
            "MAX_ITERS": "1000",
            "SKIP_CHECK": "0",
            "SAMPLED_CHECK": "1" if clean else "0",
            "SAMPLED_CHECK_M": str(BENCHMARK_SHAPE[0]),
            "SAMPLED_CHECK_N": str(BENCHMARK_SHAPE[1]),
            "SAMPLED_CHECK_K": str(BENCHMARK_SHAPE[2]),
            "SAMPLED_CHECK_TILES": "2",
            "SAMPLED_CHECK_SEED": "1",
            "SKIP_BENCH": "1" if clean else "0",
            "COUNT_BENCH_REPORTS": "0",
        }
    )
    return environment


def _parse_output(
    output: str, variant: str
) -> tuple[str | None, bool, bool, float | None]:
    architecture = None
    correctness = False
    timing = None
    device_match = re.search(r"\bgcnArch=([^,\s]+)", output)
    if device_match is not None:
        architecture = device_match.group(1)
    correctness_pattern = re.compile(
        rf"^{re.escape(variant)}\s+correctness: PASS\b", re.MULTILINE
    )
    correctness = correctness_pattern.search(output) is not None
    sampled_pattern = re.compile(
        rf"^{re.escape(variant)}\s+sampled_correctness: PASS\b", re.MULTILINE
    )
    sampled_correctness = sampled_pattern.search(output) is not None
    timing_pattern = re.compile(
        rf"^{re.escape(variant)}\s+([0-9]+(?:\.[0-9]+)?) ms\b", re.MULTILINE
    )
    timing_matches = timing_pattern.findall(output)
    if len(timing_matches) == 1:
        timing = float(timing_matches[0])
    return architecture, correctness, sampled_correctness, timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--phase", choices=("clean", "cold", "warm"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    command = [
        str(args.executable),
        "--check-m",
        str(CHECK_SHAPE[0]),
        "--check-n",
        str(CHECK_SHAPE[1]),
        "--check-k",
        str(CHECK_SHAPE[2]),
        "--bench-m",
        str(BENCHMARK_SHAPE[0]),
        "--bench-n",
        str(BENCHMARK_SHAPE[1]),
        "--bench-k",
        str(BENCHMARK_SHAPE[2]),
    ]
    variant = WORKLOADS[args.workload]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(variant, args.repetitions, args.phase),
    )
    output = completed.stdout + completed.stderr
    if output:
        print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    architecture, correctness, sampled_correctness, timing = _parse_output(
        output, variant
    )
    timing_valid = (
        timing is not None and math.isfinite(timing) and timing > 0.0
        if args.phase == "warm"
        else timing is None
    )
    sampled_oracle_valid = (
        sampled_correctness if args.phase in {"clean", "cold"} else True
    )
    accepted = (
        completed.returncode == 0
        and architecture is not None
        and architecture.startswith("gfx1201")
        and correctness
        and sampled_oracle_valid
        and timing_valid
    )
    measurement = {
        "timed_aggregate_minimum_ms": MINIMUM_TIMED_MS,
        "benchmark_repetitions": args.repetitions,
        "variant": variant,
        "check_shape": list(CHECK_SHAPE),
        "benchmark_shape": list(BENCHMARK_SHAPE),
        "oracle": "native-exact-and-production-sampled-host-reference",
        "oracle_passed": correctness and sampled_oracle_valid,
        "exact_oracle_passed": correctness,
        "sampled_oracle_passed": sampled_correctness,
        "phase": args.phase,
        "architecture": architecture,
        "returncode": completed.returncode,
    }
    if timing is not None:
        measurement.update(
            {
                "device_median_ms": timing,
                "device_samples_ms": [timing],
            }
        )
    result = {args.label: measurement}
    _write_oracle_result("pass" if accepted else "fail", result)
    print(json.dumps(result, sort_keys=True))
    if completed.returncode != 0:
        return completed.returncode
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
