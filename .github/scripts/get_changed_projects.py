#!/usr/bin/env python3
"""Get changed project paths based on git diff, validated against repos-config.json."""

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ci_utils import get_modified_paths, matches_paths, set_github_output
from config_loader import load_repo_config
from pr_detect_changed_subtrees import get_valid_prefixes, find_matched_subtrees

SCRIPT_DIR = Path(__file__).resolve().parent

SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "docs/*",
    "projects/*/docs/*",
    "shared/*/docs/*",
]

# Patterns that trigger a full test run when changed.
# This includes CI workflows and the scripts that determine project selection logic.
FULL_TEST_TRIGGER_PATTERNS = [
    ".github/workflows/therock*",
    ".github/scripts/therock*",
    ".github/scripts/get_changed_projects.py",
    ".github/scripts/ci_utils.py",
    ".github/scripts/config_loader.py",
    ".github/scripts/repo_config_model.py",
    ".github/scripts/pr_detect_changed_subtrees.py",
    ".github/repos-config.json",
]


@dataclass
class ChangedProjectsResult:
    """Result of analyzing changed projects."""

    changed_projects: str  # Comma-separated list of changed project paths
    run_all_tests: bool  # Whether to run all tests (e.g., CI files changed)
    skip_tests: bool  # Whether to skip tests entirely (e.g., docs-only changes)


def is_path_skippable(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


def should_run_all_tests(paths: Optional[Iterable[str]]) -> bool:
    """Check if any changed files should trigger a full test run."""
    if paths is None:
        return False
    return matches_paths(paths, FULL_TEST_TRIGGER_PATTERNS)


def get_changed_projects(base_ref: str) -> ChangedProjectsResult:
    """Get changed project paths validated against repos-config.json.

    Returns a ChangedProjectsResult with:
    - changed_projects: comma-separated list of changed project paths
    - run_all_tests: True if CI-related files changed (run full test suite)
    - skip_tests: True if only skippable files changed (skip tests entirely)
    """
    modified_paths = get_modified_paths(base_ref)
    if not modified_paths:
        return ChangedProjectsResult(
            changed_projects="", run_all_tests=False, skip_tests=True
        )

    # If CI workflow/script files changed, run all tests
    if should_run_all_tests(modified_paths):
        return ChangedProjectsResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

    # If only skippable files changed, skip tests
    if not check_for_non_skippable_path(modified_paths):
        return ChangedProjectsResult(
            changed_projects="", run_all_tests=False, skip_tests=True
        )

    repo_config_path = SCRIPT_DIR / ".." / "repos-config.json"
    config = load_repo_config(str(repo_config_path))
    valid_prefixes = get_valid_prefixes(config)
    matched_subtrees = find_matched_subtrees(list(modified_paths), valid_prefixes)

    return ChangedProjectsResult(
        changed_projects=",".join(matched_subtrees),
        run_all_tests=False,
        skip_tests=False,
    )


if __name__ == "__main__":
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    result = get_changed_projects(base_ref)
    set_github_output(
        {
            "changed_projects": result.changed_projects,
            "run_all_tests": str(result.run_all_tests).lower(),
            "skip_tests": str(result.skip_tests).lower(),
        }
    )
