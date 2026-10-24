#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve coco run mode, trigger kind, and CLI flags for rccl-resolve-coco-run."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import retry, set_github_output

# Must stay in step with the cron entries in rccl-coco-scheduled.yml.
WEEKDAY_NIGHTLY_CRON = "47 3 * * 1-5"
SATURDAY_CRON = "47 3 * * 6"

VALID_MODES = frozenset({"pr", "nightly", "weekly", "monthly"})

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# The coco harness file whose CLI must offer --baseline-sha before a run passes it.
HARNESS_PR_COMMAND = Path("coco") / "cli" / "commands" / "pr.py"

GitHubGet = Callable[[str], dict]


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


def build_coco_args(*, mode: str, dry_run: bool, baseline_sha: str = "") -> str:
    args = "-vv --skip-failed-builds"
    if mode == "pr":
        # No --prune-build: the build and test phases share a directory and
        # prune would delete trees the later phase still needs.
        args += " --skip-baselines"
        if baseline_sha:
            args += f" --baseline-sha {baseline_sha}"
    else:
        args += " --prune-build"
    if dry_run:
        args += " --dry-run"
    return args


def harness_supports_baseline_sha(harness_dir: Path) -> bool:
    """True when the checked-out coco harness accepts ``coco pr --baseline-sha``."""
    pr_command = harness_dir / HARNESS_PR_COMMAND
    return pr_command.is_file() and '"--baseline-sha"' in pr_command.read_text(
        encoding="utf-8"
    )


@retry(max_attempts=3, delay_seconds=2, exceptions=(urllib.error.URLError,))
def github_get(url: str) -> dict:
    """GET a GitHub REST API URL as JSON, authenticated by ``GITHUB_TOKEN``."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_merge_base(
    *, pr_repo: str, pr_number: str, pr_sha: str, get: GitHubGet = github_get
) -> str:
    """The merge-base of the PR head *pr_sha* with the PR's base branch.

    The baseline arm is built at this commit, so a PR is measured only
    against code it contains. Raises ``ValueError`` when it cannot be
    resolved, or when it is the PR head itself.
    """
    api = f"https://api.github.com/repos/{pr_repo}"
    base_ref = get(f"{api}/pulls/{pr_number}")["base"]["ref"]
    merge_base = get(f"{api}/compare/{base_ref}...{pr_sha}")["merge_base_commit"]["sha"]
    if not FULL_SHA.match(merge_base or ""):
        raise ValueError(
            f"Could not resolve the merge-base of PR #{pr_number} with {base_ref!r}."
        )
    if merge_base == pr_sha:
        raise ValueError(
            f"PR #{pr_number}'s head {pr_sha} is its merge-base with {base_ref!r}; "
            "there is no change to measure."
        )
    return merge_base


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
    baseline_sha: str = "",
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
        coco_args=build_coco_args(
            mode=mode, dry_run=dry_run, baseline_sha=baseline_sha
        ),
        pr_number=pr_number,
        pr_sha=pr_sha,
        pr_repo=pr_repo,
    )


def resolve_baseline_sha(
    *,
    pr_number: str,
    pr_sha: str,
    pr_repo: str,
    harness_dir: Path,
    get: GitHubGet = github_get,
) -> str:
    """The commit to build a PR run's baseline at, or ``""`` to keep its configured ref.

    Empty for a periodic run, and for a harness without ``--baseline-sha``,
    which is reported as a workflow warning.
    """
    if not pr_number:
        return ""
    if not harness_supports_baseline_sha(harness_dir):
        print(
            "::warning::This coco harness has no --baseline-sha; the PR is measured "
            "against the baseline's configured ref, not its merge-base."
        )
        return ""
    return resolve_merge_base(
        pr_repo=pr_repo, pr_number=pr_number, pr_sha=pr_sha, get=get
    )


def main() -> None:
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    pr_number = os.environ.get("PR_NUMBER", "")
    pr_sha = os.environ.get("PR_SHA", "")
    pr_repo = os.environ.get("PR_REPO", "")
    try:
        resolved = resolve_run(
            pr_number=pr_number,
            pr_sha=pr_sha,
            pr_repo=pr_repo,
            event_name=os.environ.get("EVENT_NAME", ""),
            schedule=os.environ.get("SCHEDULE", ""),
            periodic_mode=os.environ.get("PERIODIC_MODE", "nightly"),
            dry_run=dry_run,
            baseline_sha=resolve_baseline_sha(
                pr_number=pr_number,
                pr_sha=pr_sha,
                pr_repo=pr_repo,
                harness_dir=Path.cwd(),
            ),
        )
    except (ValueError, urllib.error.URLError) as exc:
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
