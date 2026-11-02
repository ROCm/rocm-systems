#!/usr/bin/env python3
"""Compute RCCL CI outputs for the top-level GitHub Actions workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

# Test policy lives here; workflows only handle dependencies and invoke suites.
# Multi-node rccl-tests remains disabled until cluster qualification is complete.
SUITES = {
    "single-node": ("gfx94X-dcgpu", False),
    "rocprof": ("gfx94X-dcgpu", False),
    "pytorch": ("gfx94X-dcgpu", True),
    "jax": ("gfx94X-dcgpu", True),
    "madengine": ("gfx950-dcgpu", True),
    "accl-profiler": ("gfx950-dcgpu", True),
}
FAMILIES = ("gfx94X-dcgpu", "gfx950-dcgpu")


def select_suites(event_name: str, selection: str = "auto") -> list[str]:
    if selection == "auto":
        scheduled = event_name in {"schedule", "workflow_dispatch"}
        return [
            name for name, (_, nightly) in SUITES.items() if scheduled or not nightly
        ]
    names = [name.strip() for name in selection.split(",")]
    if any(name not in SUITES for name in names):
        raise ValueError(
            "test_suites must be auto or a comma-separated list of: "
            + ", ".join(SUITES)
        )
    return list(dict.fromkeys(names))


def families_for(selection: str, suites: list[str]) -> list[str]:
    # Preserve both builds for the default policy, even when gfx950 tests are
    # disabled on presubmit. Explicit selections build only what they consume.
    return [
        family
        for family in FAMILIES
        if selection == "auto" or any(SUITES[name][0] == family for name in suites)
    ]


def runnable_suites(
    event_name: str,
    selection: str,
    family: str,
    build_result: str,
    artifact_run_id: str,
) -> list[str]:
    suites = select_suites(event_name, selection)
    if family not in FAMILIES:
        raise ValueError(f"Unsupported AMDGPU family: {family}")
    # A skipped build is valid only for explicit artifact reuse.
    ready = build_result == "success" or (
        build_result == "skipped" and bool(artifact_run_id)
    )
    return [name for name in suites if ready and SUITES[name][0] == family]


def changed_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def plan(event_name: str, test_scope: str, now: datetime) -> tuple[bool, str, str]:
    if event_name in {"schedule", "workflow_dispatch"}:
        scope = "all" if event_name == "schedule" and now.weekday() == 5 else test_scope
        return True, scope, "Scheduled/manual run — always run CI"

    files = changed_files()
    runs_ci = any(
        path.startswith(("projects/rccl/", ".github/workflows/therock-rccl-"))
        for path in files
    )
    return (
        runs_ci,
        test_scope,
        "RCCL changes detected" if runs_ci else "No RCCL changes detected",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--test-scope", choices=("smoke", "all"), default="smoke")
    parser.add_argument("--test-suites", default="auto")
    parser.add_argument("--amdgpu-family", choices=FAMILIES)
    parser.add_argument(
        "--build-result",
        choices=("success", "skipped", "failure", "cancelled"),
        default="success",
    )
    parser.add_argument("--artifact-run-id", default="")
    args = parser.parse_args()
    try:
        suites = select_suites(args.event_name, args.test_suites)
        if args.amdgpu_family:
            outputs = {
                "suites": runnable_suites(
                    args.event_name,
                    args.test_suites,
                    args.amdgpu_family,
                    args.build_result,
                    args.artifact_run_id,
                )
            }
        else:
            runs_ci, scope, message = plan(
                args.event_name, args.test_scope, datetime.now(timezone.utc)
            )
            print(message)
            outputs = {
                "run_linux_ci": runs_ci,
                "test_scope": scope,
                "amdgpu_families": families_for(args.test_suites, suites),
            }
    except ValueError as exc:
        parser.error(str(exc))
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for name, value in outputs.items():
            rendered = (
                value
                if isinstance(value, str)
                else json.dumps(value, separators=(",", ":"))
            )
            print(f"{name}={rendered}")
            stream.write(f"{name}={rendered}\n")


if __name__ == "__main__":
    main()
