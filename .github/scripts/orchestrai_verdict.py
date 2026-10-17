#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve one suite's verdict from a batch pipeline build.

A batch build runs several suites on one machine, so the build's overall result
cannot tell us whether THIS suite passed. The pipeline writes a per-group rollup
to its summary.json:

    groups["suite-<id>"] = {passed, failed, skipped, errors, status}

keyed by the submission group id (independent of completion order). This script
prefers that rollup and falls back to the build console's per-suite lifecycle if
the rollup is absent.

Usage:
    orchestrai_verdict.py     (all inputs via env)

Env:
    BUILD_URL, ORCHESTRAI_PIPELINE_USER, ORCHESTRAI_PIPELINE_TOKEN,
    SUITE_ID, PLATFORM, DEVICE

Outputs:
    test-results/summary.json         per-suite counts, for artifact upload
    rp_url=<reportportal launch url>  -> $GITHUB_OUTPUT
    exit 0 if PASSED, 1 otherwise (including: no verdict found for this suite)
"""

import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

COUNT_KEYS = ("passed", "failed", "skipped", "errors")
FETCH_TIMEOUT = 60


def fetch(url: str, user: str, token: str) -> str:
    """GET `url` with basic auth, returning "" on any failure.

    SECURITY: the credentials are deliberately NOT handed to a `curl -u
    user:token` subprocess. argv is world-readable via /proc/<pid>/cmdline and
    `ps` for the lifetime of the process, so on a shared self-hosted runner any
    other local process could read the pipeline token. Doing the HTTP in-process
    (stdlib only — this job has no pip install step) keeps the secret in this
    process's memory only.
    """
    req = urllib.request.Request(url)
    req.add_header(
        "Authorization",
        "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return ""


def status_from_group(group: dict) -> str | None:
    """PASSED/FAILED from a group rollup, or None when it carries no verdict
    (no status field and zero counts — i.e. nothing actually ran)."""
    status = group.get("status")
    if status:
        return status
    counts = {k: group.get(k) or 0 for k in COUNT_KEYS}
    if sum(counts.values()) == 0:
        return None  # ambiguous: the group exists but ran nothing — not a PASS
    return "FAILED" if (counts["failed"] or counts["errors"]) else "PASSED"


def status_from_console(console: str, suite_id: str) -> str | None:
    """Fallback: the console exposes the per-suite lifecycle even when
    summary.json has no group for it:
        Started suite "suite-<id> #<n>" ... id=<sid>
        Finished item <sid> -> PASSED|FAILED
    """
    started = re.search(
        rf'Started suite "suite-{re.escape(suite_id)} #\d+".*?id=([0-9a-fA-F-]+)',
        console,
    )
    if not started:
        return None
    # The separator between the item id and the status renders differently across
    # console encodings (Unicode "→", ASCII "->", or mojibake if the log was
    # decoded wrong). Match any run of non-alphanumerics rather than one arrow.
    finished = re.findall(
        rf"Finished item {re.escape(started.group(1))}[^A-Za-z0-9]+([A-Z]+)", console
    )
    return finished[-1] if finished else None


def main() -> None:
    user = os.environ.get("ORCHESTRAI_PIPELINE_USER", "")
    token = os.environ.get("ORCHESTRAI_PIPELINE_TOKEN", "")
    build_url = os.environ.get("BUILD_URL", "")
    suite_id = os.environ.get("SUITE_ID", "")
    platform = os.environ.get("PLATFORM", "")
    device = os.environ.get("DEVICE", "")

    if not suite_id or not platform:
        print(
            "::error::orchestrai_verdict.py: SUITE_ID and PLATFORM must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    group_key = f"suite-{suite_id}"
    results_dir = Path("test-results")
    results_dir.mkdir(exist_ok=True)

    def write_summary(ok: bool, counts: dict | None = None) -> None:
        (results_dir / "summary.json").write_text(
            json.dumps(
                {
                    "suite_id": suite_id,
                    "platform": platform,
                    "device": device,
                    "passed": 1 if ok else 0,
                    "failed": 0 if ok else 1,
                    "counts": counts or {},
                },
                indent=2,
            )
        )

    if not build_url:
        write_summary(False)
        print(f"::error::No pipeline build URL for {suite_id} ({platform}/{device})")
        sys.exit(1)

    raw = fetch(f"{build_url}artifact/.pipeline/summary.json", user, token) or "{}"
    try:
        summary = json.loads(raw)
    except ValueError:
        summary = {}

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"rp_url={summary.get('rp_launch_url', '')}\n")

    status, how = None, ""
    group = (summary.get("groups") or {}).get(group_key)
    if isinstance(group, dict):
        status = status_from_group(group)
        if status:
            how = "summary.json groups"

    if status is None:
        console = fetch(f"{build_url}consoleText", user, token)
        status = status_from_console(console, suite_id)
        if status:
            how = "console suite lifecycle"

    if status is None:
        write_summary(False)
        print(
            f"::error::No verdict for {suite_id} ({platform}/{device}): no "
            f"groups['{group_key}'] in summary.json and no suite result in the "
            f"console. groups present: {list((summary.get('groups') or {}).keys())}"
        )
        sys.exit(1)

    counts = {k: group.get(k) for k in COUNT_KEYS} if isinstance(group, dict) else {}
    ok = status == "PASSED"
    write_summary(ok, counts)
    print(
        f"{suite_id} ({platform}/{device}): {'PASS' if ok else 'FAIL'} "
        f"(status={status}, via {how}, counts={counts})"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
