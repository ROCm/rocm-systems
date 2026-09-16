#!/usr/bin/env python3
"""RCCL subtree freeze: block develop PRs that touch projects/rccl*.

Driven by .github/workflows/rccl-subtree-freeze.yml. Reads PR metadata
through the GitHub API and never checks out PR code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

FROZEN_PREFIXES = ("projects/rccl", "projects/rccl-tests")
# GitHub's pull-files REST endpoint returns at most 3000 files.
MAX_LISTED_FILES = 3000
CHECK_NAME = "rccl-subtree-freeze"
SWITCH_ENV = "RCCL_SUBTREE_FREEZE_ENABLED"


def gh(*args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(["gh", "api", *args], capture_output=True, text=True, input=stdin)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh api {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def gh_paginate(path: str) -> list[Any]:
    """--paginate emits one JSON array per page; flatten them into one list."""
    out = gh("--paginate", path)
    decoder = json.JSONDecoder()
    items: list[Any] = []
    while out := out.lstrip():
        page, end = decoder.raw_decode(out)
        items.extend(page)
        out = out[end:]
    return items


def freeze_enabled() -> bool:
    return (os.environ.get(SWITCH_ENV) or "true").strip().lower() not in {"false", "off", "0"}


def is_frozen(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in FROZEN_PREFIXES)


def evaluate(repo: str, pr: int) -> tuple[bool, str]:
    """Return (allowed, summary). Anything unverifiable fails closed."""
    if not freeze_enabled():
        return True, f"RCCL subtree freeze is off ({SWITCH_ENV}={os.environ.get(SWITCH_ENV)})."

    changed = json.loads(gh(f"repos/{repo}/pulls/{pr}")).get("changed_files")
    if changed is None:
        return False, "PR metadata omitted changed_files; failing closed."
    if changed > MAX_LISTED_FILES:
        return False, (
            f"PR changes {changed} files; GitHub lists at most {MAX_LISTED_FILES}. Failing closed."
        )

    files = gh_paginate(f"repos/{repo}/pulls/{pr}/files")
    if len(files) < changed:
        return False, f"Listed {len(files)} files but PR reports {changed}. Failing closed."

    blocked = [
        path
        for entry in files
        for path in (entry.get("filename"), entry.get("previous_filename"))
        if path and is_frozen(path)
    ]
    if blocked:
        listing = "\n".join(f"- `{path}`" for path in blocked)
        return False, (
            "This PR cannot land: it modifies frozen trees `projects/rccl` and/or "
            f"`projects/rccl-tests`.\n\nBlocked paths:\n\n{listing}"
        )
    return True, "No frozen RCCL subtree paths were changed."


def post_check(repo: str, head_sha: str, allowed: bool, summary: str) -> None:
    payload = {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "success" if allowed else "failure",
        "output": {"title": CHECK_NAME, "summary": summary},
    }
    gh("--method", "POST", f"repos/{repo}/check-runs", "--input", "-", stdin=json.dumps(payload))


def report(text: str) -> None:
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")


def cmd_evaluate(repo: str, pr: int) -> int:
    allowed, summary = evaluate(repo, pr)
    report(summary)
    if allowed:
        return 0
    print(f"::error::{summary.splitlines()[0]}")
    return 1


def cmd_reevaluate_open(repo: str, base: str) -> int:
    """Re-post the check on open PRs so idle ones follow the current switch."""
    pulls = gh_paginate(f"repos/{repo}/pulls?base={base}&state=open&per_page=100")
    lines = [f"Freeze enabled: {freeze_enabled()}. Re-checked {len(pulls)} open `{base}` PR(s)."]
    for pull in pulls:
        number, head_sha = pull["number"], pull["head"]["sha"]
        try:
            allowed, summary = evaluate(repo, number)
        except Exception as exc:
            allowed, summary = False, f"Failed to evaluate PR #{number}; failing closed: {exc}"
        try:
            post_check(repo, head_sha, allowed, summary)
        except Exception as exc:
            lines.append(f"- #{number}: could not post check ({exc})")
            continue
        lines.append(f"- #{number} (`{head_sha[:12]}`): {'pass' if allowed else 'FAIL'}")
    # Blocked PRs carry their own failing check run, so this job still succeeds.
    report("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "reevaluate-open"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--base", default="develop")
    args = parser.parse_args()

    if args.command == "reevaluate-open":
        return cmd_reevaluate_open(args.repo, args.base)
    if args.pr is None:
        parser.error("evaluate requires --pr")
    return cmd_evaluate(args.repo, args.pr)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"::error::{exc}")
        sys.exit(1)
