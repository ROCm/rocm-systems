#!/usr/bin/env python3
"""Evaluate the RCCL subtree freeze for a PR, or re-apply it to open PRs.

Used by .github/workflows/rccl-subtree-freeze.yml. Does not check out PR
code; it only reads PR metadata through the GitHub API.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Iterable, List, Sequence

FROZEN_PREFIXES = ("projects/rccl", "projects/rccl-tests")
# GitHub's pull-files REST endpoint returns at most 3000 files.
MAX_LISTED_FILES = 3000
CHECK_NAME = "rccl-subtree-freeze"


def freeze_is_enabled(value: str | None) -> bool:
    switch = (value or "true").strip().lower()
    return switch not in {"false", "off", "0"}


def is_frozen_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in FROZEN_PREFIXES)


def gh_json(args: Sequence[str]) -> Any:
    result = subprocess.run(
        ["gh", "api", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh api {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def gh_paginate_json_arrays(path: str) -> List[Any]:
    result = subprocess.run(
        ["gh", "api", "--paginate", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh api --paginate {path} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    decoder = json.JSONDecoder()
    data = result.stdout.strip()
    items: List[Any] = []
    idx = 0
    while idx < len(data):
        while idx < len(data) and data[idx].isspace():
            idx += 1
        if idx >= len(data):
            break
        parsed, offset = decoder.raw_decode(data, idx)
        idx = offset
        if isinstance(parsed, list):
            items.extend(parsed)
        else:
            items.append(parsed)
    return items


def collect_paths(files: Iterable[dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for entry in files:
        filename = entry.get("filename") or ""
        previous = entry.get("previous_filename") or ""
        if filename:
            paths.append(filename)
        if previous:
            paths.append(previous)
    return paths


def evaluate_pr(repo: str, pr_number: int, enabled: bool) -> tuple[bool, str]:
    if not enabled:
        return True, (
            "RCCL subtree freeze is off "
            f"(RCCL_SUBTREE_FREEZE_ENABLED={os.environ.get('RCCL_SUBTREE_FREEZE_ENABLED')})."
        )

    pr = gh_json([f"repos/{repo}/pulls/{pr_number}"])
    changed_files = pr.get("changed_files")
    if changed_files is None:
        return False, "Pull request metadata omitted changed_files; failing closed."
    if changed_files > MAX_LISTED_FILES:
        return False, (
            f"PR changes {changed_files} files; GitHub lists at most "
            f"{MAX_LISTED_FILES}. Failing closed."
        )

    files = gh_paginate_json_arrays(f"repos/{repo}/pulls/{pr_number}/files")
    if len(files) < changed_files:
        return False, (
            f"Listed {len(files)} files but PR reports {changed_files}. Failing closed."
        )

    blocked = [path for path in collect_paths(files) if is_frozen_path(path)]
    if blocked:
        lines = [
            "This PR cannot land: it modifies frozen trees `projects/rccl` and/or `projects/rccl-tests`.",
            "",
            "Blocked paths:",
            "",
            *[f"- `{path}`" for path in blocked],
        ]
        return False, "\n".join(lines)

    return True, "No frozen RCCL subtree paths were changed."


def post_check_run(repo: str, head_sha: str, success: bool, summary: str) -> None:
    payload = {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "status": "completed",
        "conclusion": "success" if success else "failure",
        "output": {
            "title": "RCCL subtree freeze" if not success else "RCCL subtree freeze passed",
            "summary": summary,
        },
    }
    result = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repo}/check-runs",
            "--input",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        input=json.dumps(payload),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to post check run for {head_sha}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def append_step_summary(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def cmd_evaluate(repo: str, pr_number: int) -> int:
    enabled = freeze_is_enabled(os.environ.get("RCCL_SUBTREE_FREEZE_ENABLED"))
    ok, summary = evaluate_pr(repo, pr_number, enabled)
    print(summary)
    append_step_summary(summary if ok else f"### RCCL subtree freeze\n\n{summary}")
    if not ok:
        print(f"::error::{summary.splitlines()[0]}")
        return 1
    return 0


def cmd_reevaluate_open(repo: str, base: str) -> int:
    enabled = freeze_is_enabled(os.environ.get("RCCL_SUBTREE_FREEZE_ENABLED"))
    pulls = gh_paginate_json_arrays(f"repos/{repo}/pulls?base={base}&state=open&per_page=100")
    lines = [
        f"Re-evaluating {len(pulls)} open PR(s) targeting `{base}`.",
        f"Freeze enabled: {enabled}.",
        "",
    ]
    for pr in pulls:
        number = pr["number"]
        head_sha = pr["head"]["sha"]
        try:
            ok, summary = evaluate_pr(repo, number, enabled)
            post_check_run(repo, head_sha, ok, summary)
        except Exception as exc:
            ok = False
            summary = f"Failed to evaluate PR #{number}; failing closed: {exc}"
            try:
                post_check_run(repo, head_sha, False, summary)
            except Exception as post_exc:
                lines.append(f"- #{number}: could not post check ({post_exc})")
        status = "pass" if ok else "FAIL"
        lines.append(f"- #{number} (`{head_sha[:12]}`): {status}")
        print(f"PR #{number}: {status}\n{summary}\n")

    report = "\n".join(lines)
    print(report)
    append_step_summary(report)
    # The develop-push job should succeed even when some PRs fail the freeze;
    # those failures are recorded as check runs on the PR head SHAs.
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--repo", required=True)
    evaluate.add_argument("--pr", required=True, type=int)

    reevaluate = sub.add_parser("reevaluate-open")
    reevaluate.add_argument("--repo", required=True)
    reevaluate.add_argument("--base", default="develop")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return cmd_evaluate(args.repo, args.pr)
    return cmd_reevaluate_open(args.repo, args.base)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"::error::{exc}")
        sys.exit(1)
