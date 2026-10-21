#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide what the RCCL perf regression gate should do for a pull request.

Emits for the `changes` job in rccl_perf_regression.yml:
  run_gate=true|false   false only when there is a non-doc PERF_PATH_PATTERNS
                        change and every one is covered by a COVERAGE_RULES
                        workflow on the base branch.
  build_rccl=1|0        1 when anything under projects/rccl/ changed; 0 runs the
                        A/B against the fixed library instead.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import get_modified_paths, matches_paths, set_github_output
from rccl_change_sets import RCCL_PATTERNS, covered_elsewhere, is_doc, is_rccl

# fnmatch twin of rccl_perf_regression.yml's pull_request paths (tested).
PERF_PATH_PATTERNS = (
    *RCCL_PATTERNS,
    ".github/workflows/rccl_perf_regression.yml",
    ".github/scripts/rccl_detect_perf_changes.py",
    ".github/scripts/rccl_change_sets.py",
)


def should_run_gate(changed_files: Iterable[str], base_branch: str = "") -> bool:
    """See run_gate in the module docstring."""
    code = [
        f
        for f in changed_files
        if matches_paths([f], PERF_PATH_PATTERNS) and not is_doc(f)
    ]
    return not (code and all(covered_elsewhere(f, base_branch) for f in code))


def should_build_rccl(changed_files: Iterable[str]) -> bool:
    """Whether to build reference and candidate librccl from the PR."""
    return any(is_rccl(f) for f in changed_files)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether the RCCL perf regression gate runs for a PR."
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Reference to diff against, `HEAD^` for a PR merge commit.",
    )
    parser.add_argument(
        "--base-branch",
        default="",
        help="PR target branch; selects which COVERAGE_RULES apply.",
    )
    args = parser.parse_args(argv)

    # A file moved into a covered path must still count as removed from its old one.
    changed_files = get_modified_paths(args.base_ref, no_renames=True)
    run = should_run_gate(changed_files, args.base_branch)
    build = should_build_rccl(changed_files)
    logging.info("Perf gate on %r: run=%s build=%s", args.base_branch, run, build)
    set_github_output(
        {"run_gate": "true" if run else "false", "build_rccl": "1" if build else "0"}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
