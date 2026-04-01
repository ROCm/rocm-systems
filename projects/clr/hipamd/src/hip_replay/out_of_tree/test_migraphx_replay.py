#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License
#
# Integration test: record ONNX light models with HRR then replay with hrr-bench.
#
# Exit codes:
#   0  - all models passed (or test skipped due to missing dependencies)
#   1  - one or more models failed
#
# Usage:
#   python3 test_migraphx_replay.py \
#     --hrr-bench   /path/to/hrr-bench \
#     --libhrr      /path/to/libhrr_record.so \
#     --record-script /path/to/record_model.py
#
# Skips (exit 0) when:
#   - libhrr_record.so not built
#   - hrr-bench binary not found
#   - migraphx Python module not importable
#   - onnx light models not installed

import argparse
import os
import re
import subprocess
import sys
import tempfile


MODELS = ["light_resnet50", "light_squeezenet", "light_densenet121"]

# Sanity bounds: median kernel time must be in (0, 10000) ms.
MEDIAN_MIN_MS = 0.0
MEDIAN_MAX_MS = 10000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HRR MIGraphX ONNX replay integration test")
    p.add_argument("--hrr-bench", required=True, metavar="PATH",
                   help="Path to hrr-bench binary")
    p.add_argument("--libhrr", required=True, metavar="PATH",
                   help="Path to libhrr_record.so")
    p.add_argument("--record-script", required=True, metavar="PATH",
                   help="Path to record_model.py")
    p.add_argument("--iterations", type=int, default=20,
                   help="hrr-bench app --iterations (default: 20)")
    p.add_argument("--warmup", type=int, default=5,
                   help="hrr-bench app --warmup (default: 5)")
    return p.parse_args()


def skip(reason: str) -> None:
    print(f"SKIP: {reason}", flush=True)
    sys.exit(0)


def find_light_model_dir() -> str:
    """Return path to onnx backend light test data directory."""
    try:
        import onnx
        light = os.path.join(os.path.dirname(onnx.__file__),
                             "backend", "test", "data", "light")
        if os.path.isdir(light):
            return light
    except ImportError:
        pass
    return ""


def run_test(name: str, light_dir: str, args: argparse.Namespace,
             workdir: str) -> tuple[bool, str]:
    """Record and replay one model. Returns (passed, message)."""
    onnx_path = os.path.join(light_dir, f"{name}.onnx")
    capture_dir = os.path.join(workdir, f"{name}.hrr")

    if not os.path.isfile(onnx_path):
        return False, f"ONNX model not found: {onnx_path}"

    # --- Phase 1: compile (no LD_PRELOAD) ---
    print(f"  [{name}] compiling...", flush=True)
    try:
        subprocess.run(
            [sys.executable, args.record_script, "--compile", onnx_path],
            check=True, timeout=180,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return False, f"compile failed (exit {e.returncode})"
    except subprocess.TimeoutExpired:
        return False, "compile timed out"

    # --- Phase 2: record one inference ---
    print(f"  [{name}] recording...", flush=True)
    env = os.environ.copy()
    env["LD_PRELOAD"] = args.libhrr
    env["HRR_RECORD"] = "1"
    env["HRR_OUTPUT"] = capture_dir
    env["HRR_RECORD_MODE"] = "inputs"
    try:
        subprocess.run(
            [sys.executable, args.record_script, "--record", onnx_path],
            env=env, check=True, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return False, f"record failed (exit {e.returncode})"
    except subprocess.TimeoutExpired:
        return False, "record timed out"

    if not os.path.isdir(capture_dir):
        return False, "capture directory not created"

    # --- Phase 3: hrr-bench app ---
    print(f"  [{name}] replaying ({args.iterations} iters, {args.warmup} warmup)...",
          flush=True)
    try:
        result = subprocess.run(
            [args.hrr_bench, "app", capture_dir,
             "--iterations", str(args.iterations),
             "--warmup", str(args.warmup)],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "hrr-bench timed out"

    combined = result.stdout + result.stderr

    if result.returncode != 0:
        return False, (f"hrr-bench exited {result.returncode}\n"
                       + combined[-500:])  # last 500 chars for context

    if "GPU fault" in combined or "unspecified launch failure" in combined:
        return False, "GPU fault detected in output"

    # Parse median and unit from output (e.g. "Median:     0.506 ms")
    m = re.search(r"Median:\s+([\d.]+)\s+(ms|us)", combined)
    if not m:
        return False, "could not parse Median from hrr-bench output"

    val, unit = float(m.group(1)), m.group(2)
    median_ms = val if unit == "ms" else val / 1000.0

    if not (MEDIAN_MIN_MS < median_ms < MEDIAN_MAX_MS):
        return False, f"median {median_ms:.3f} ms out of expected range"

    # Extract kernel count for the summary line
    kc = re.search(r"Kernels in trace:\s+(\d+)", combined)
    kernel_count = kc.group(1) if kc else "?"

    return True, f"median={median_ms:.3f}ms  kernels={kernel_count}"


def main() -> None:
    args = parse_args()

    # --- Dependency checks (skip rather than fail if tools/libs missing) ---
    if not os.path.isfile(args.libhrr):
        skip(f"libhrr_record.so not found: {args.libhrr}")

    if not os.path.isfile(args.hrr_bench):
        skip(f"hrr-bench not found: {args.hrr_bench}")

    try:
        import migraphx  # noqa: F401
    except ImportError:
        skip("migraphx Python module not available")

    light_dir = find_light_model_dir()
    if not light_dir:
        skip("onnx light models not found (install onnx package)")

    print(f"HRR MIGraphX replay test — {len(MODELS)} models", flush=True)
    print(f"  hrr-bench:     {args.hrr_bench}", flush=True)
    print(f"  libhrr:        {args.libhrr}", flush=True)
    print(f"  light models:  {light_dir}", flush=True)
    print(flush=True)

    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="hrr_test_") as workdir:
        for name in MODELS:
            passed, msg = run_test(name, light_dir, args, workdir)
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {name}  {msg}", flush=True)
            if not passed:
                failures.append(f"{name}: {msg}")

    print(flush=True)
    if failures:
        print(f"FAILED: {len(failures)}/{len(MODELS)} model(s):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)

    print(f"PASSED: all {len(MODELS)} models", flush=True)


if __name__ == "__main__":
    main()
