#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repeat a workload with deterministic ConSan offsets; never join run evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


SELECTORS = (
    "RUNTIME_SAMPLE_STRIDE", "RUNTIME_SAMPLE_OFFSET",
    "WORKGROUP_SAMPLE_STRIDE", "WORKGROUP_SAMPLE_OFFSET",
    "CELL_SAMPLE_STRIDE", "CELL_SAMPLE_OFFSET", "SAMPLED_BANKS",
)
PREFIX = "RJ_CONSAN_"
FIELDS = re.compile(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)")


def stride(value: str) -> int:
    result = int(value)
    if result < 1 or result > 1 << 24 or result & (result - 1):
        raise argparse.ArgumentTypeError("stride must be a power of two in 1..16777216")
    return result


def offsets(value: str) -> list[int]:
    try:
        result = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("offsets must be comma-separated integers") from error
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("offsets must be nonnegative and distinct")
    return result


def schedule(wg_stride: int, cell_stride: int, wg: list[int], cell: list[int], budget: int):
    if budget < 1 or budget > 4096:
        raise ValueError("run budget must be in 1..4096")
    if max(wg) >= wg_stride or max(cell) >= cell_stride:
        raise ValueError("each offset must be smaller than its associated stride")
    return list(itertools.islice(itertools.product(wg, cell), budget))


def parse_log(path: Path) -> dict:
    result = {"configurations": [], "reports": [], "verdicts": [], "examples": [],
              "geometry": [], "coverage": []}
    markers = (
        ("ConSan configuration ", "configurations"),
        ("ConSan conflict ", "examples"),
        ("ConSan auto report reader=", "reports"),
        ("ConSan auto report plan ", "geometry"),
        ("ConSan auto report buffer ", "geometry"),
        ("ConSan diagnostic map ", "geometry"),
        ("ConSan analysis verdict ", "verdicts"),
        ("ConSan coverage ", "coverage"),
    )
    with path.open(errors="replace") as stream:
        for line in stream:
            for marker, key in markers:
                if marker in line:
                    result[key].append(dict(FIELDS.findall(line)))
                    break
    # This is a sum of reported counters, not a claim about unique racing
    # dynamic accesses. Examples can be capped even when this sum is nonzero.
    result["reported_conflict_pairs"] = sum(
        int(row.get("conflicts", "0")) for row in result["reports"])
    result["evidence_observed"] = bool(
        result["configurations"] and result["reports"] and result["verdicts"])
    return result


def diagnostic_catalog(runs: list[dict]) -> list[dict]:
    catalog = {}
    for run in runs:
        for index, example in enumerate(run["evidence"]["examples"]):
            sites = tuple(sorted((tuple(example.get(f"{side}_{key}") for key in
                                       ("instruction", "kind", "bytes"))
                                  for side in ("first", "second")), key=repr))
            identity = (example.get("code_object"), sites)
            if not identity[0] or any(site[0] in (None, "unavailable", "ambiguous") for site in sites):
                identity += (run["run"], index)  # unresolved sites cannot coalesce across runs
            entry = catalog.setdefault(identity, {"code_object": identity[0], "sites": sites,
                                                   "observations": []})
            # Every reference points to an already diagnosed pair in ONE run.
            entry["observations"].append({"run": run["run"], "example": index})
    return list(catalog.values())


def execute(command: list[str], env: dict[str, str], log: Path, timeout: float) -> tuple[int, bool, float]:
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, timed_out, time.monotonic() - start


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workgroup-stride", type=stride, default=256)
    parser.add_argument("--cell-stride", type=stride, default=256)
    parser.add_argument("--workgroup-offsets", type=offsets, default=[0])
    parser.add_argument("--cell-offsets", type=offsets, default=[0])
    parser.add_argument("--banks", type=int, choices=(0, 1, 2, 4, 8), default=0)
    parser.add_argument("--run-budget", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not args.hook.is_file() or not 0 < args.timeout < float("inf"):
        parser.error("provide a command, existing hook file and finite positive timeout")
    try:
        selected = schedule(args.workgroup_stride, args.cell_stride, args.workgroup_offsets,
                            args.cell_offsets, args.run_budget)
    except ValueError as error:
        parser.error(str(error))
    executable = shutil.which(command[0])
    if executable is None:
        parser.error(f"command executable not found: {command[0]}")
    if args.output.exists():
        parser.error("output directory already exists; use a new directory for this investigation")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    for name in SELECTORS:
        env.pop(PREFIX + name, None)
    env.update(HSA_TOOLS_LIB=str(args.hook.resolve()), RJ_CONSAN_MODE="default",
               RJ_CONSAN_LOG="1", RJ_CONSAN_FAIL_CLOSED="1", RJ_CONSAN_REQUIRE_PATCH="1")
    env.update({PREFIX + "WORKGROUP_SAMPLE_STRIDE": str(args.workgroup_stride),
                PREFIX + "CELL_SAMPLE_STRIDE": str(args.cell_stride),
                PREFIX + "SAMPLED_BANKS": str(args.banks)})
    manifest = {
        "command": command, "cwd": str(Path.cwd()),
        "hook": str(args.hook.resolve()),
        "hashes": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.hook, Path(executable))},
        "environment": {k: v for k, v in env.items() if k.startswith(
            ("RJ_", "ROCM", "HIP", "HSA_", "LD_", "ROCR_"))},
        "schedule": selected, "requested_combinations": len(args.workgroup_offsets) * len(args.cell_offsets),
        "run_budget": args.run_budget, "timeout_seconds": args.timeout,
        "order": "workgroup offsets outer, cell offsets inner, in command-line order",
    }
    write_json(args.output / "manifest.json", manifest)
    runs = []
    failed = False
    for index, (wg, cell) in enumerate(selected):
        current = dict(env, **{PREFIX + "WORKGROUP_SAMPLE_OFFSET": str(wg),
                              PREFIX + "CELL_SAMPLE_OFFSET": str(cell)})
        log = args.output / f"run-{index:04d}.log"
        code, timed_out, elapsed = execute(command, current, log, args.timeout)
        evidence = parse_log(log)
        row = {"run": index, "workgroup_offset": wg, "cell_offset": cell,
               "returncode": code, "timed_out": timed_out, "elapsed_seconds": elapsed,
               "log": log.name, "evidence": evidence}
        runs.append(row)
        write_json(args.output / f"run-{index:04d}.json", row)
        failed = code != 0 or timed_out or not evidence["evidence_observed"]
        write_json(args.output / "summary.json", {
            "runs": runs, "diagnostic_catalog": diagnostic_catalog(runs),
            "completed_schedule": len(runs) == len(selected) and not failed,
            "stopped_on_failure": failed,
            "interpretation": "Each example is a pair diagnosed within one execution. "
                              "No diagnostic observed does not mean race-free. "
                              "Check completeness, retention and correctness per run.",
        })
        print(f"run={index} wg_offset={wg} cell_offset={cell} "
              f"reported_pairs={evidence['reported_conflict_pairs']} returncode={code}", flush=True)
        if failed:
            break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
