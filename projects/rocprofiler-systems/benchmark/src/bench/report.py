# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""JSON export of the raw measurements and the comparison."""

from __future__ import annotations

import json
import platform
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench import system
from bench.runner import RunResult

# Deny-list, not allow-list: a new RunResult field is exported without touching this file.
# name and index are implied by the position in commands[name]["runs"].
OMITTED_RUN_FIELDS = {"name", "index"}


def env_info() -> dict[str, str | None]:
    return {
        "cpu": system.cpu_info()["model"],
        "kernel": platform.release(),
    }


def limitations(results: dict[str, list[RunResult]], meta: dict[str, Any]) -> list[str]:
    runs = [r for rs in results.values() for r in rs]
    notes = [
        (
            "wall-clock includes scheduling and blocking; CPU time covers "
            "only the launched process (RUSAGE of the waited child), not "
            "other processes the tool may spawn"
        )
    ]
    if runs and all(r.t1_t2_ns is None for r in runs):
        notes.append("t0_t1/t1_t2/t2_t3 are null: no BENCH_FD hook markers received")
    if not meta.get("mem_sample_ms"):
        notes.append("sampled RSS disabled; only peak RSS from wait4 reported")
    notes.append("GPU time and per-thread CPU are not measured")
    if len(runs) and min(len(rs) for rs in results.values()) < 100:
        notes.append("fewer than 100 runs: p95 is not stable")
    return notes


def build_json(
    results: dict[str, list[RunResult]],
    commands: dict[str, str],
    analysis: dict[str, Any],
    meta: dict[str, Any],
    envs: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """envs: per-command variables added to os.environ; recorded as commands[n]["env"]."""
    envs = envs or {}
    return {
        "meta": {**meta, "env": env_info()},
        "commands": {
            n: {
                "command": shlex.split(commands[n]),
                "env": envs.get(n, {}),
                "runs": [
                    {k: v for k, v in asdict(r).items() if k not in OMITTED_RUN_FIELDS}
                    for r in results[n]
                ],
            }
            for n in results
        },
        "summary": analysis["summary"],
        "comparison": analysis["comparison"],
        "checks": {"drift": analysis["drift"], "warnings": analysis["warnings"]},
        "limitations": limitations(results, meta),
        "errors": analysis["errors"],
    }


def write_json(doc: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
