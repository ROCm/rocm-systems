#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide whether a pull request needs the RCCL host unit tests.

Emits ``rccl=true|false`` for the `changes` job in rccl-host-unit-tests.yml.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import get_modified_paths, matches_paths, set_github_output
from rccl_change_sets import RCCL_PATTERNS

# fnmatch patterns: `*` also spans `/`.
HOST_TEST_PATH_PATTERNS = [
    *RCCL_PATTERNS,
    # Files that decide whether the tests run.
    ".github/workflows/rccl-host-unit-tests.yml",
    ".github/scripts/rccl_detect_host_test_changes.py",
    ".github/scripts/rccl_change_sets.py",
    ".github/scripts/ci_utils.py",
]


def touches_host_tests(changed_files: Iterable[str]) -> bool:
    """Whether the host unit tests must run for this file list."""
    return matches_paths(changed_files, HOST_TEST_PATH_PATTERNS)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Detect whether a PR needs the RCCL host unit tests."
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="Reference to diff against, `HEAD^` for a PR merge commit.",
    )
    args = parser.parse_args(argv)

    changed_files = get_modified_paths(args.base_ref, no_renames=True)
    touched = touches_host_tests(changed_files)
    set_github_output({"rccl": "true" if touched else "false"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
