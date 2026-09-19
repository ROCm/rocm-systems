#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve coco run mode, trigger kind, and CLI flags for resolve-coco-run."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import set_github_output

# Must stay in step with the cron entries in rccl-coco-scheduled.yml.
WEEKDAY_NIGHTLY_CRON = "47 3 * * 1-5"
SATURDAY_CRON = "47 3 * * 6"

VALID_MODES = frozenset({"pr", "nightly", "weekly", "monthly"})


@dataclass(frozen=True)
class ResolvedRun:
    mode: str
    trigger_kind: str
    coco_ref: str
    coco_args: str
    pr_number: str
    pr_sha: str
    pr_repo: str


def schedule_to_mode(schedule: str, day_of_month: int) -> str:
    """Map the cron expression that fired to nightly, weekly, or monthly."""
    if schedule == WEEKDAY_NIGHTLY_CRON:
        return "nightly"
    if schedule == SATURDAY_CRON:
        # cron has no nth-weekday syntax, so "3rd Saturday" is a day-of-month range.
        if 15 <= day_of_month <= 21:
            return "monthly"
        return "weekly"
    raise ValueError(
        f"Unrecognised schedule {schedule!r}. "
        "Add it alongside the cron entry in rccl-coco-scheduled.yml."
    )


def resolve_mode_and_trigger(
    *,
    pr_number: str,
    event_name: str,
    schedule: str,
    periodic_mode: str,
    day_of_month: int | None = None,
) -> tuple[str, str]:
    """Return (mode, trigger_kind) for this workflow event."""
    if pr_number:
        return "pr", "ci"
    if event_name == "schedule":
        if day_of_month is None:
            day_of_month = datetime.now(timezone.utc).day
        return schedule_to_mode(schedule, day_of_month), "scheduled"
    return periodic_mode, "manual"


def build_coco_args(*, mode: str, dry_run: bool) -> str:
    args = "-vv --skip-failed-builds"
    if mode == "pr":
        # No --prune-build: the build and test phases share a directory and
        # prune would delete trees the later phase still needs.
        args += " --skip-baselines"
    else:
        args += " --prune-build"
    if dry_run:
        args += " --dry-run"
    return args


def resolve_run(
    *,
    pr_number: str,
    pr_sha: str,
    pr_repo: str,
    event_name: str,
    schedule: str,
    periodic_mode: str,
    dry_run: bool,
    day_of_month: int | None = None,
    coco_sha: str | None = None,
) -> ResolvedRun:
    mode, trigger_kind = resolve_mode_and_trigger(
        pr_number=pr_number,
        event_name=event_name,
        schedule=schedule,
        periodic_mode=periodic_mode,
        day_of_month=day_of_month,
    )
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}; expected pr, nightly, weekly or monthly."
        )
    if coco_sha is None:
        coco_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return ResolvedRun(
        mode=mode,
        trigger_kind=trigger_kind,
        coco_ref=coco_sha,
        coco_args=build_coco_args(mode=mode, dry_run=dry_run),
        pr_number=pr_number,
        pr_sha=pr_sha,
        pr_repo=pr_repo,
    )


def main() -> None:
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    try:
        resolved = resolve_run(
            pr_number=os.environ.get("PR_NUMBER", ""),
            pr_sha=os.environ.get("PR_SHA", ""),
            pr_repo=os.environ.get("PR_REPO", ""),
            event_name=os.environ.get("EVENT_NAME", ""),
            schedule=os.environ.get("SCHEDULE", ""),
            periodic_mode=os.environ.get("PERIODIC_MODE", "nightly"),
            dry_run=dry_run,
        )
    except ValueError as exc:
        print(f"::error::{exc}")
        raise SystemExit(1) from exc

    set_github_output(
        {
            "mode": resolved.mode,
            "trigger_kind": resolved.trigger_kind,
            "coco_ref": resolved.coco_ref,
            "coco_args": resolved.coco_args,
            "pr_number": resolved.pr_number,
            "pr_sha": resolved.pr_sha,
            "pr_repo": resolved.pr_repo,
        }
    )


if __name__ == "__main__":
    main()
