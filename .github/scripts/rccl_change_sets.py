# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared path criteria for RCCL CI gating.

Facts about changed paths only: no git, argparse or GITHUB_OUTPUT. Each gate
(rccl_detect_coco_changes.py, rccl_detect_perf_changes.py) decides what to do
with them.

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

    The examples drive the tests: every rule must show what it covers and what
    it must not. Only `workflow`'s trigger is tested, not whether branch
    protection requires it.
    """

    name: str
    workflow: str
    branches: tuple[str, ...]
    patterns: tuple[str, ...]
    covered_examples: tuple[str, ...]
    lookalike_examples: tuple[str, ...]


COVERAGE_RULES = (
    CoverageRule(
        name="host-only unit tests",
        workflow=".github/workflows/rccl-host-unit-tests.yml",
        branches=("develop",),
        patterns=("projects/rccl/test/host/*",),
        covered_examples=(
            "projects/rccl/test/host/CMakeLists.txt",
            "projects/rccl/test/host/run_host_tests.sh",
            "projects/rccl/test/host/fakes/os_fakes.cc",
        ),
        lookalike_examples=(
            "projects/rccl/test/host",
            "projects/rccl/test/hostile.cc",
            "projects/rccl/test/hostfoo/x.cc",
            "projects/rccl/test/CMakeLists.txt",
            "projects/rccl/src/test/host/x.cc",
            "projects/rccl-tests/test/host/x.cc",
        ),
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
