#!/usr/bin/env python3
"""Compute RCCL CI outputs for the top-level GitHub Actions workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone


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
    runs_ci = any(path.startswith("projects/rccl/") for path in files)
    return runs_ci, test_scope, "RCCL changes detected" if runs_ci else "No RCCL changes detected"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--test-scope", required=True)
    args = parser.parse_args()
    runs_ci, scope, message = plan(args.event_name, args.test_scope, datetime.now(timezone.utc))
    print(message)
    output = os.environ["GITHUB_OUTPUT"]
    with open(output, "a", encoding="utf-8") as stream:
        stream.write(f"run_linux_ci={str(runs_ci).lower()}\n")
        stream.write(f"test_scope={scope}\n")


if __name__ == "__main__":
    main()
