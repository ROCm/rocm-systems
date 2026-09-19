#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard RCCL cluster scale-runner usage in GitHub Actions workflows.

Runners and permitted workflows live in .github/scripts/allowlist/
cluster-runners.allowlist. The same workflow list applies to every listed
runner. On pull requests, newly added references are flagged so CI can request
@ROCm/rccl-ci review.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import set_github_output

ALLOWLIST_PATH = Path(".github/scripts/allowlist/cluster-runners.allowlist")
WORKFLOWS_DIR = Path(".github/workflows")
# Runner labels also live here; usage is allowed when every workflow that
# reads the matrix output is on the allowlist.
MATRIX_SOURCES = {
    Path(".github/scripts/rccl_coco_matrix.py"): {
        ".github/workflows/rccl-coco-pr.yml",
        ".github/workflows/rccl-coco-scheduled.yml",
    },
}


def load_allowlist(path: Path) -> tuple[set[str], set[str]]:
    runners: set[str] = set()
    workflows: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(".github/workflows/"):
            workflows.add(line)
        elif line.startswith(".github/"):
            # Anything else under .github/ is scanned as a path, never matched
            # as one, so it would sit in the file doing nothing.
            raise ValueError(
                f"{path}: only .github/workflows/ paths are allowed: {line}"
            )
        else:
            runners.add(line)
    return runners, workflows


def line_assigns_runner(line: str, runner: str) -> bool:
    """Whether this line references ``runner`` anywhere, not only under a key.

    A label reaches a job through ``runs-on:``, a YAML block-sequence item, a
    matrix entry and a workflow_call input default, so any line may carry one.
    The boundaries stop an allowlisted label from vouching for a longer one,
    so `...-scale-runner` does not permit `...-scale-runner-v2`. Runner labels
    are case-insensitive, so this comparison is too.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    pattern = rf"(?<![\w-]){re.escape(runner)}(?![\w-])"
    return re.search(pattern, stripped, re.IGNORECASE) is not None


def _scan_file_for_runners(path: Path, runners: set[str]) -> dict[str, set[str]]:
    hits = {runner: set() for runner in runners}
    rel = path.as_posix()
    for raw in path.read_text(encoding="utf-8").splitlines():
        for runner in runners:
            if line_assigns_runner(raw, runner):
                hits[runner].add(rel)
    return hits


def scan_workflow_usage(runners: set[str]) -> dict[str, set[str]]:
    usage = {runner: set() for runner in runners}
    for path in sorted(WORKFLOWS_DIR.iterdir()):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        for runner, paths in _scan_file_for_runners(path, runners).items():
            usage[runner].update(paths)
    for path in sorted(MATRIX_SOURCES):
        if not path.is_file():
            continue
        for runner, paths in _scan_file_for_runners(path, runners).items():
            usage[runner].update(paths)
    return usage


def matrix_source_allowed(
    path: str, allowed_workflows: set[str]
) -> bool:
    consumers = MATRIX_SOURCES.get(Path(path))
    if consumers is None:
        return False
    return consumers.issubset(allowed_workflows)


def git_diff_new_runner_usage(
    base: str, head: str, runners: set[str]
) -> dict[str, list[tuple[str, str]]]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            f"{base}..{head}",
            "--",
            ".github/workflows",
            ".github/scripts/rccl_coco_matrix.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    additions: dict[str, list[tuple[str, str]]] = {runner: [] for runner in runners}
    current_file: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
        elif line.startswith("+") and not line.startswith("+++"):
            content = line[1:]
            for runner in runners:
                if line_assigns_runner(content, runner) and current_file:
                    additions[runner].append((current_file, content.strip()))
    return additions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Check RCCL cluster scale-runner workflow allowlist."
    )
    parser.add_argument(
        "--base",
        help="Base git ref for PR diff (omit on workflow_dispatch).",
    )
    parser.add_argument(
        "--head",
        help="Head git ref for PR diff (omit on workflow_dispatch).",
    )
    args = parser.parse_args(argv)

    # The pair drives the new-reference check. Supplying one alone falls
    # through to a full-tree scan only, which reports no new references and
    # exits 0.
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be given together, or both omitted.")

    if not ALLOWLIST_PATH.is_file():
        print(f"Missing allowlist: {ALLOWLIST_PATH}", file=sys.stderr)
        return 1

    runners, allowed_workflows = load_allowlist(ALLOWLIST_PATH)
    if not runners:
        print(f"No runners listed in {ALLOWLIST_PATH}.", file=sys.stderr)
        return 1
    if not allowed_workflows:
        print(f"No workflows listed in {ALLOWLIST_PATH}.", file=sys.stderr)
        return 1

    exit_code = 0

    for runner, using in sorted(scan_workflow_usage(runners).items()):
        disallowed = sorted(
            path
            for path in using
            if path not in allowed_workflows
            and not matrix_source_allowed(path, allowed_workflows)
        )
        if not disallowed:
            continue
        exit_code = 1
        print(
            f"Error: {runner} appears in workflow(s) not on the allowlist:",
            file=sys.stderr,
        )
        for path in disallowed:
            print(f"  {path}", file=sys.stderr)
        print(
            f"Add the workflow to {ALLOWLIST_PATH} or remove the runner reference.",
            file=sys.stderr,
        )

    any_new_lines = False
    if args.base and args.head:
        for runner, new_lines in git_diff_new_runner_usage(
            args.base, args.head, runners
        ).items():
            if not new_lines:
                continue
            any_new_lines = True
            print(f"New {runner} reference(s) in this PR:")
            for path, content in new_lines:
                print(f"  {path}: {content}")

            new_in_disallowed = sorted(
                {
                    path
                    for path, _ in new_lines
                    if path not in allowed_workflows
                    and not matrix_source_allowed(path, allowed_workflows)
                }
            )
            if new_in_disallowed:
                exit_code = 1
                print(
                    f"Error: new {runner} usage in workflow(s) not on the allowlist:",
                    file=sys.stderr,
                )
                for path in new_in_disallowed:
                    print(f"  {path}", file=sys.stderr)

    set_github_output({"new_runner_lines": "true" if any_new_lines else "false"})
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
