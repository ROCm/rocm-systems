#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def parse_env_assignment(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"empty environment key in {text!r}")
    return key, value


def apply_env(base, assignments):
    env = dict(base)
    for key, value in assignments:
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    return env


def run_once(command, env, timeout):
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed_us = int((time.perf_counter() - start) * 1000000)
    return {
        "elapsed_us": elapsed_us,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def summarize(samples):
    elapsed = [sample["elapsed_us"] for sample in samples]
    if not elapsed:
        return {
            "runs": 0,
            "median_us": 0,
            "min_us": 0,
            "max_us": 0,
            "mean_us": 0,
        }
    return {
        "runs": len(elapsed),
        "median_us": int(statistics.median(elapsed)),
        "min_us": min(elapsed),
        "max_us": max(elapsed),
        "mean_us": int(statistics.mean(elapsed)),
    }


def fail_run(phase, sample):
    sys.stderr.write(f"{phase} run failed with exit {sample['returncode']}\n")
    if sample["stdout"]:
        sys.stderr.write("--- stdout ---\n")
        sys.stderr.write(sample["stdout"])
        if not sample["stdout"].endswith("\n"):
            sys.stderr.write("\n")
    if sample["stderr"]:
        sys.stderr.write("--- stderr ---\n")
        sys.stderr.write(sample["stderr"])
        if not sample["stderr"].endswith("\n"):
            sys.stderr.write("\n")


def run_phase(name, command, env, warmups, iterations, timeout):
    warmup_samples = []
    for _ in range(warmups):
        sample = run_once(command, env, timeout)
        if sample["returncode"] != 0:
            fail_run(f"{name} warmup", sample)
            raise SystemExit(1)
        warmup_samples.append(sample)

    samples = []
    for _ in range(iterations):
        sample = run_once(command, env, timeout)
        if sample["returncode"] != 0:
            fail_run(name, sample)
            raise SystemExit(1)
        samples.append(sample)
    return warmup_samples, samples


def compact_sample(sample):
    out = {
        "elapsed_us": sample["elapsed_us"],
        "returncode": sample["returncode"],
    }
    stdout = sample["stdout"].strip()
    stderr = sample["stderr"].strip()
    if stdout:
        out["stdout_last_line"] = stdout.splitlines()[-1]
    if stderr:
        out["stderr_last_line"] = stderr.splitlines()[-1]
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure rocfuzz preload runtime overhead by comparing the same "
            "host command with baseline and instrumented environments."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="runtime-overhead")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--env", action="append", default=[],
                        type=parse_env_assignment)
    parser.add_argument("--baseline-env", action="append", default=[],
                        type=parse_env_assignment)
    parser.add_argument("--instrumented-env", action="append", default=[],
                        type=parse_env_assignment)
    parser.add_argument("--max-overhead-ratio-x1000", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    common_env = apply_env(os.environ, args.env)
    baseline_env = apply_env(common_env, args.baseline_env)
    instrumented_env = apply_env(common_env, args.instrumented_env)

    baseline_warmups, baseline_samples = run_phase(
        "baseline", command, baseline_env, args.warmups, args.iterations,
        args.timeout)
    instrumented_warmups, instrumented_samples = run_phase(
        "instrumented", command, instrumented_env, args.warmups,
        args.iterations, args.timeout)

    baseline_summary = summarize(baseline_samples)
    instrumented_summary = summarize(instrumented_samples)
    baseline_median = baseline_summary["median_us"]
    instrumented_median = instrumented_summary["median_us"]
    overhead_delta_us = instrumented_median - baseline_median
    overhead_ratio_x1000 = 0
    if baseline_median > 0:
        overhead_ratio_x1000 = int((instrumented_median * 1000) / baseline_median)

    report = {
        "label": args.label,
        "command": command,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "baseline": {
            **baseline_summary,
            "samples": [compact_sample(sample) for sample in baseline_samples],
            "warmup_samples": [
                compact_sample(sample) for sample in baseline_warmups
            ],
        },
        "instrumented": {
            **instrumented_summary,
            "samples": [
                compact_sample(sample) for sample in instrumented_samples
            ],
            "warmup_samples": [
                compact_sample(sample) for sample in instrumented_warmups
            ],
        },
        "overhead_delta_us": overhead_delta_us,
        "overhead_ratio_x1000": overhead_ratio_x1000,
        "threshold_enforced": args.max_overhead_ratio_x1000 > 0,
        "max_overhead_ratio_x1000": args.max_overhead_ratio_x1000,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        f"{args.label}: baseline median {baseline_median} us, "
        f"instrumented median {instrumented_median} us, "
        f"ratio x1000 {overhead_ratio_x1000}"
    )

    if (args.max_overhead_ratio_x1000 > 0 and
            overhead_ratio_x1000 > args.max_overhead_ratio_x1000):
        sys.stderr.write(
            f"overhead ratio {overhead_ratio_x1000} exceeds threshold "
            f"{args.max_overhead_ratio_x1000}\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
