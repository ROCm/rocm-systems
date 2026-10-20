#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Classify a pull request's changed files for RCCL CI gating.

Emits ``rccl=true|false`` for the `changes` job in rccl-coco-pr.yml, and
``host_tests_only=true|false`` for the one in rccl_perf_regression.yml.
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

# fnmatch patterns: `*` also spans `/`.
COCO_PATH_PATTERNS = [
    "projects/rccl/*",
    "projects/rccl-tests/*",
    ".github/workflows/rccl-coco-pr.yml",
    ".github/workflows/rccl-coco-scheduled.yml",
    ".github/workflows/rccl-coco-run.yml",
    ".github/scripts/rccl_coco_matrix.py",
    ".github/scripts/rccl_resolve_coco_run.py",
    ".github/actions/rccl-resolve-coco-run/*",
    ".github/actions/rccl-checkout-coco-harness/*",
    ".github/scripts/rccl_detect_coco_changes.py",
    ".github/scripts/ci_utils.py",
]

# Subset of get_changed_projects.py's SKIPPABLE_PATH_PATTERNS, duplicated rather
# than imported because that module pulls in pydantic.
SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "projects/*/docs/*",
]

# Host-only tests, which coco never runs. rccl-host-unit-tests.yml runs them,
# but only on PRs to HOST_TEST_BRANCHES, so they are skipped only there.
HOST_TEST_PATH_PATTERNS = [
    "projects/rccl/test/host/*",
]
HOST_TEST_BRANCHES = ("develop",)

_MAX_LOGGED_PATHS = 50


def coco_paths(changed_files: Iterable[str], base_branch: str = "") -> list[str]:
    """The subset of ``changed_files`` that the coco suites cover."""
    skippable = SKIPPABLE_PATH_PATTERNS
    if base_branch in HOST_TEST_BRANCHES:
        skippable = skippable + HOST_TEST_PATH_PATTERNS
    return [
        f
        for f in changed_files
        if matches_paths([f], COCO_PATH_PATTERNS) and not matches_paths([f], skippable)
    ]


def host_tests_only(changed_files: Iterable[str], base_branch: str = "") -> bool:
    """Whether every non-doc change is a host-only test on HOST_TEST_BRANCHES."""
    if base_branch not in HOST_TEST_BRANCHES:
        return False
    code = [f for f in changed_files if not matches_paths([f], SKIPPABLE_PATH_PATTERNS)]
    return bool(code) and all(matches_paths([f], HOST_TEST_PATH_PATTERNS) for f in code)


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
        help="PR target branch; host-only tests are skipped only on "
        + ", ".join(HOST_TEST_BRANCHES)
        + ".",
    )
    args = parser.parse_args(argv)

    # A file moved into an excluded path must still count as removed from its old one.
    changed_files = get_modified_paths(args.base_ref, no_renames=True)
    touched = touches_coco(changed_files, args.base_branch)
    host_only = host_tests_only(changed_files, args.base_branch)
    logging.info("Host-only test changes on %r: %s", args.base_branch, host_only)
    set_github_output(
        {
            "rccl": "true" if touched else "false",
            "host_tests_only": "true" if host_only else "false",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
