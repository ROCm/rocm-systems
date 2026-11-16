#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide whether a pull request touches anything the RCCL coco suites test.

Emits ``rccl=true|false`` for the `changes` job in rccl-coco-pr.yml.
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
from rccl_change_sets import RCCL_PATTERNS, covered_elsewhere, is_doc

# fnmatch patterns: `*` also spans `/`.
COCO_PATH_PATTERNS = [
    *RCCL_PATTERNS,
    "projects/rccl-tests/*",
    ".github/workflows/rccl-coco-pr.yml",
    ".github/workflows/rccl-coco-scheduled.yml",
    ".github/workflows/rccl-coco-run.yml",
    ".github/scripts/rccl_coco_matrix.py",
    ".github/scripts/rccl_resolve_coco_run.py",
    ".github/actions/rccl-resolve-coco-run/*",
    ".github/actions/rccl-checkout-coco-harness/*",
    ".github/scripts/rccl_detect_coco_changes.py",
    ".github/scripts/rccl_change_sets.py",
    ".github/scripts/ci_utils.py",
]

_MAX_LOGGED_PATHS = 50


def coco_paths(changed_files: Iterable[str], base_branch: str = "") -> list[str]:
    """The subset of ``changed_files`` that the coco suites cover."""
    return [
        f
        for f in changed_files
        if matches_paths([f], COCO_PATH_PATTERNS)
        and not is_doc(f)
        and not covered_elsewhere(f, base_branch)
    ]


def touches_coco(changed_files: Iterable[str], base_branch: str = "") -> bool:
    """Whether the coco suites must run for this file list."""
    changed_files = list(changed_files)
    matched = coco_paths(changed_files, base_branch)
    if not matched:
        logging.info("No coco paths among %d changed files.", len(changed_files))
        return False

    logging.info("Coco paths touched (%d):", len(matched))
    for path in sorted(matched)[:_MAX_LOGGED_PATHS]:
        logging.info("  %s", path)
    if len(matched) > _MAX_LOGGED_PATHS:
        logging.info("  ... and %d more", len(matched) - _MAX_LOGGED_PATHS)
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Detect whether a PR touches the RCCL coco test surface."
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
    touched = touches_coco(changed_files, args.base_branch)
    set_github_output({"rccl": "true" if touched else "false"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
