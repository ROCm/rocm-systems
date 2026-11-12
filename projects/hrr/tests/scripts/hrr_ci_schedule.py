#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Schedule budget for an HRR GPU integration job.

GitHub's job timeout starts when a runner picks the job up, so a job can sit
queued well past that limit. The dispatcher polls this helper: once the GPU
job is in progress it must be waited out (a deadlock is still a failure), and
if it is still waiting when the budget expires the suite is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys

# Matches the budget documented on the integration dispatcher.
SCHEDULE_LIMIT_S = 30 * 60

_SCHEDULED = frozenset({"in_progress", "completed"})


def schedule_action(
    elapsed_s: float,
    job_status: str | None,
    run_status: str | None = None,
    limit_s: int = SCHEDULE_LIMIT_S,
) -> str:
    """Return ``run``, ``skip``, or ``poll``.

    ``run`` means the GPU job was accepted by a runner, or the dispatched run
    already finished, so the caller must propagate that conclusion. ``skip``
    means the budget expired while the job was still waiting for a runner.
    """
    status = job_status or ""
    if status in _SCHEDULED or run_status == "completed":
        return "run"
    if elapsed_s >= limit_s:
        return "skip"
    return "poll"


def choose_dispatched_run(runs, title: str, seen_ids) -> str:
    """Id of the GPU run this dispatch created.

    Re-running failed jobs keeps the parent ``github.run_id``, so the previous
    attempt's GPU run is still listed. ``seen_ids`` are the runs that already
    had this title before the dispatch; their artifact must not be reused.
    """
    seen = {str(item) for item in seen_ids if str(item)}
    chosen = ""
    chosen_at = ""
    for run in runs:
        if not isinstance(run, dict) or run.get("event") != "workflow_dispatch":
            continue
        if title not in (run.get("displayTitle"), run.get("name")):
            continue
        run_id = str(run.get("databaseId") or "")
        if not run_id or run_id in seen:
            continue
        created = str(run.get("createdAt") or "")
        if not chosen or created >= chosen_at:
            chosen = run_id
            chosen_at = created
    return chosen


def skip_junit(message: str) -> str:
    """One skipped case, so the family table shows a skip rather than a pass."""
    escaped = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<testsuites>\n"
        '  <testsuite name="hrr-integration" tests="1" failures="0" '
        'errors="0" skipped="1">\n'
        '    <testcase classname="hrr" name="integration suite">\n'
        f'      <skipped message="{escaped}"/>\n'
        "    </testcase>\n"
        "  </testsuite>\n"
        "</testsuites>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elapsed", type=float)
    parser.add_argument("--status", default="")
    parser.add_argument("--run-status", default="")
    parser.add_argument("--limit", type=int, default=SCHEDULE_LIMIT_S)
    parser.add_argument("--write-skip")
    parser.add_argument("--message")
    parser.add_argument("--choose-run", action="store_true")
    parser.add_argument("--title", default="")
    parser.add_argument("--seen", default="")
    args = parser.parse_args(argv)

    if args.choose_run:
        if not args.title:
            parser.error("--choose-run requires --title")
        raw = sys.stdin.read()
        runs = json.loads(raw) if raw.strip() else []
        print(choose_dispatched_run(runs, args.title, args.seen.split()))
        return 0

    if args.write_skip is not None:
        if not args.message:
            parser.error("--write-skip requires --message")
        with open(args.write_skip, "w", encoding="utf-8") as handle:
            handle.write(skip_junit(args.message))
        return 0

    if args.elapsed is None:
        parser.error("--elapsed is required unless --write-skip is set")
    print(
        schedule_action(
            args.elapsed,
            args.status or None,
            args.run_status or None,
            args.limit,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
