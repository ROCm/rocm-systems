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


def _environment(
    variant: str,
    repetitions: int,
    phase: str,
    fixed_iterations: int | None,
    minimum_timed_ms: float,
) -> dict[str, str]:
    clean = phase == "clean"
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
            "MIN_MS": str(minimum_timed_ms),
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
    if fixed_iterations is not None:
        environment["BENCH_FIXED_ITERS"] = str(fixed_iterations)
    return environment


def _parse_output(
    output: str, variant: str
) -> tuple[str | None, bool, bool, float | None, int | None, float | None, int]:
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
    aggregate_pattern = re.compile(
        rf"^{re.escape(variant)}\s+benchmark_iterations=([0-9]+) "
        rf"benchmark_aggregate_ms=([0-9]+(?:\.[0-9]+)?)\b",
        re.MULTILINE,
    )
    aggregate_matches = aggregate_pattern.findall(output)
    iterations = int(aggregate_matches[0][0]) if len(aggregate_matches) == 1 else None
    aggregate_ms = (
        float(aggregate_matches[0][1]) if len(aggregate_matches) == 1 else None
    )
    return (
        architecture,
        correctness,
        sampled_correctness,
        timing,
        iterations,
        aggregate_ms,
        len(timing_matches),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    parser.add_argument("--phase", choices=("clean", "warm"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--fixed-iterations", type=int)
    parser.add_argument("--minimum-timed-ms", type=float, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.fixed_iterations is not None and args.fixed_iterations <= 0:
        parser.error("--fixed-iterations must be positive")
    if not math.isfinite(args.minimum_timed_ms) or args.minimum_timed_ms <= 0.0:
        parser.error("--minimum-timed-ms must be finite and positive")

    command = [str(args.executable)]
    variant = WORKLOADS[args.workload]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(
            variant,
            args.repetitions,
            args.phase,
            args.fixed_iterations,
            args.minimum_timed_ms,
        ),
    )
    output = completed.stdout + completed.stderr
    if output:
        print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    (
        architecture,
        correctness,
        sampled_correctness,
        timing,
        iterations,
        aggregate_ms,
        timing_match_count,
    ) = _parse_output(output, variant)
    timing_valid = (
        timing is not None
        and math.isfinite(timing)
        and timing > 0.0
        and iterations is not None
        and aggregate_ms is not None
        and math.isfinite(aggregate_ms)
        and aggregate_ms >= args.minimum_timed_ms
        and (args.fixed_iterations is None or iterations == args.fixed_iterations)
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
    rejection_reasons = []
    if completed.returncode != 0:
        rejection_reasons.append(f"process returncode={completed.returncode}")
    if architecture is None or not architecture.startswith("gfx1201"):
        rejection_reasons.append(f"architecture={architecture!r}, expected gfx1201")
    if not correctness:
        rejection_reasons.append("exact correctness oracle failed")
    if not sampled_oracle_valid:
        rejection_reasons.append("sampled production-shape oracle failed")
    if not timing_valid:
        rejection_reasons.append(
            f"invalid timing evidence: matches={timing_match_count}, "
            f"iterations={iterations}, aggregate_ms={aggregate_ms}"
        )
    measurement = {
        "timed_aggregate_minimum_ms": args.minimum_timed_ms,
        "benchmark_repetitions": args.repetitions,
        "variant": variant,
        "check_shape": list(CHECK_SHAPE),
        "benchmark_shape": list(BENCHMARK_SHAPE),
        "oracle": "native-exact-and-production-sampled-host-reference",
        "oracle_passed": correctness and sampled_oracle_valid,
        "exact_oracle_passed": correctness,
        "sampled_oracle_passed": sampled_correctness,
        "sampled_oracle_applicable": args.phase == "clean",
        "phase": args.phase,
        "architecture": architecture,
        "returncode": completed.returncode,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "timing_match_count": timing_match_count,
    }
    if timing is not None:
        measurement.update(
            {
                "device_median_ms": timing,
                "device_samples_ms": [timing],
                "benchmark_iterations": iterations,
                "timed_aggregate_ms": aggregate_ms,
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
