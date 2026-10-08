#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run pre-commit on PR changes or all checked-out emulation files."""

import os
from pathlib import Path
import subprocess

FORMATTING_POLICY_FILES = {
    ".pre-commit-config.yaml",
    ".github/workflows/rocjitsu-formatting.yml",
    ".github/scripts/rocjitsu_formatting.py",
}


def select_files(base_sha: str, head_sha: str) -> list[str]:
    # HEAD can be GitHub's merge commit; use the PR head for the diff.
    base = subprocess.check_output(
        ["git", "merge-base", base_sha, head_sha], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "-z", base, head_sha]
    )
    files = {os.fsdecode(path) for path in changed.split(b"\0") if path}
    if files & FORMATTING_POLICY_FILES:
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "emulation/"])
        files.update(os.fsdecode(path) for path in tracked.split(b"\0") if path)

    # Sparse checkout determines which tracked components are available to check.
    return sorted(path for path in files if Path(path).is_file())


def main() -> int:
    files = select_files(os.environ["BASE_SHA"], os.environ["HEAD_SHA"])
    if files:
        print(f"Checking {len(files)} files", flush=True)
        return subprocess.run(["pre-commit", "run", "--files", *files]).returncode
    print("No checked-in files to check in sparse checkout, skipping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
