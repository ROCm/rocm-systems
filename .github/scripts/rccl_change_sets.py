# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared path criteria for RCCL CI gating.

Facts about changed paths only: no git, argparse or GITHUB_OUTPUT.
rccl_detect_coco_changes.py decides what to do with them.

All patterns are fnmatch: `*` also spans `/`. This is not the glob dialect of a
workflow's `paths:` block, where `*` stops at `/` and `**` spans it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ci_utils import matches_paths

RCCL_PATTERNS = ("projects/rccl/*",)

# Subset of get_changed_projects.py's SKIPPABLE_PATH_PATTERNS, duplicated rather
# than imported because that module pulls in pydantic.
DOC_PATTERNS = (
    "*.md",
    "*.rst",
    "projects/*/docs/*",
)


@dataclass(frozen=True)
class CoverageRule:
    """Paths that `workflow` already checks on PRs to `branches`.

    tests/rccl_change_sets_test.py needs covered and look-alike examples for
    every rule. It checks `workflow`'s trigger, not whether branch protection
    requires it.
    """

    name: str
    workflow: str
    branches: tuple[str, ...]
    patterns: tuple[str, ...]


COVERAGE_RULES = (
    # Builds and runs test/host standalone. The in-tree BUILD_TESTS=ON build of
    # test/host (and its CMake) is only compiled by TheRock Multi-Arch CI.
    CoverageRule(
        name="host-only unit tests",
        workflow=".github/workflows/rccl-host-unit-tests.yml",
        branches=("develop",),
        patterns=(
            "projects/rccl/test/host/*",
            # CTest categories of the micro host binaries. Only test/host's in-tree
            # CMake reads them; the GPU suites' test_categories*.yaml do not match.
            "projects/rccl/test/test_categories_micro*.yaml",
        ),
    ),
    # Runs the test_runner unit tests. coco's PR suites are harness-owned YAML
    # and never run the test runner; math-ci precheckin runs its configs.
    CoverageRule(
        name="test runner",
        workflow=".github/workflows/rccl-host-unit-tests.yml",
        branches=("develop",),
        patterns=("projects/rccl/tools/scripts/test_runner/*",),
    ),
)


def is_rccl(path: str) -> bool:
    return matches_paths([path], RCCL_PATTERNS)


def is_doc(path: str) -> bool:
    return matches_paths([path], DOC_PATTERNS)


def covered_elsewhere(path: str, base_branch: str) -> bool:
    """Whether a COVERAGE_RULES workflow already checks ``path`` on ``base_branch``."""
    return any(
        base_branch in rule.branches and matches_paths([path], rule.patterns)
        for rule in COVERAGE_RULES
    )
