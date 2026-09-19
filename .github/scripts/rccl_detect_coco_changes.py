#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide whether a pull request touches anything the RCCL coco suites test.

Emits ``rccl=true|false`` for the `changes` job in rccl-coco-pr.yml. A false lets
that workflow's gate pass a PR without running anything on the clusters.
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

# fnmatch patterns, in which `*` also spans `/`. The CI files that decide how
# the gate runs are included, so a change to one is exercised by the PR making
# it.
COCO_PATH_PATTERNS = [
    "projects/rccl/*",
    "projects/rccl-tests/*",
    ".github/workflows/rccl-coco-pr.yml",
    ".github/workflows/rccl-coco-scheduled.yml",
    ".github/workflows/rccl-coco-run.yml",
    ".github/scripts/rccl_coco_matrix.py",
    ".github/actions/resolve-coco-run/*",
    ".github/actions/checkout-coco-harness/*",
    ".github/scripts/rccl_detect_coco_changes.py",
    ".github/scripts/ci_utils.py",
]

# Doc-only paths cannot change a build or a test result. This is the subset of
# get_changed_projects.py's SKIPPABLE_PATH_PATTERNS that can overlap the
# patterns above; that module is not imported because it pulls in pydantic.
SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "projects/*/docs/*",
]

_MAX_LOGGED_PATHS = 50


def coco_paths(changed_files: Iterable[str]) -> list[str]:
    """The subset of ``changed_files`` that the coco suites cover."""
    return [
        f
        for f in changed_files
        if matches_paths([f], COCO_PATH_PATTERNS)
        and not matches_paths([f], SKIPPABLE_PATH_PATTERNS)
    ]


def touches_coco(changed_files: Iterable[str]) -> bool:
    """Whether the coco suites must run for this file list."""
    changed_files = list(changed_files)
    matched = coco_paths(changed_files)
    if not matched:
        logging.info("No RCCL paths among %d changed files.", len(changed_files))
        return False

    logging.info("RCCL paths touched (%d):", len(matched))
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
    args = parser.parse_args(argv)

    changed_files = get_modified_paths(args.base_ref)
    set_github_output({"rccl": "true" if touches_coco(changed_files) else "false"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
