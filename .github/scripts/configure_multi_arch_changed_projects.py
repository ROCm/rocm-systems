#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure per-PR project/test selection for rocm-systems multi-arch CI.

This is the rocm-systems-owned entry point for the multi-arch CI `configure`
job. It computes the ``changed_projects`` / ``run_all_tests`` / ``skip_tests``
outputs consumed by TheRock's ``setup_multi_arch.yml`` and by
``compute_build_stages.py``.

Why this lives in rocm-systems (and not in TheRock's configure_external_repo_ci.py):
``repos-config.json`` only lists the subtree-synced ``projects/*`` repos, so the
in-monorepo directories ``shared/*`` (amdgpu-windows-interop, kpack,
machine-readable-isa) and ``emulation/*`` (mirage, rocjitsu) are not subtree
repos and are invisible to a repos-config-only prefix match. Without surfacing
them, a PR confined to those paths yields an empty ``changed_projects`` and
TheRock falls back to building and testing everything. rocm-systems owns which
of its own directories are CI-relevant, so the surfacing list and the
unclassified-change guard live here.

The script is intentionally dependency-free (stdlib + the ``gh`` CLI, which is
present on GitHub runners) so the `configure` job only needs a sparse checkout
of this file and ``repos-config.json`` -- no pydantic, no full checkout.

Usage:
    python configure_multi_arch_changed_projects.py \
        --event-name pull_request \
        --github-repo ROCm/rocm-systems \
        --base-sha <sha> --head-sha <sha> \
        --config-path .github/repos-config.json

Outputs (to $GITHUB_OUTPUT):
    changed_projects : comma-separated subtree paths (e.g. "projects/rdc")
    run_all_tests    : "true" when CI-infra / unclassified changes force a full run
    skip_tests       : "true" when only docs/skippable files changed
"""

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Files that never require tests on their own.
SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "docs/*",
    "projects/*/docs/*",
    "shared/*/docs/*",
]

# Changes that force a full test run (CI infrastructure + shared test harness).
FULL_TEST_TRIGGER_PATTERNS = [
    ".github/workflows/therock*",
    ".github/scripts/therock*",
    ".github/scripts/configure_multi_arch_changed_projects.py",
    ".github/scripts/compute_build_stages.py",
    ".github/scripts/ci_utils.py",
    ".github/scripts/config_loader.py",
    ".github/scripts/repo_config_model.py",
    ".github/scripts/pr_detect_changed_subtrees.py",
    ".github/repos-config.json",
    # shared/ctest holds the CTest categorization logic consumed by every
    # project's tests; a change there can alter selection everywhere, so treat
    # it as a full-test trigger rather than a single surfaced component.
    "shared/ctest/*",
]

# CI-relevant monorepo directories that are NOT subtree-synced repos and so are
# absent from repos-config.json. Without these, a PR confined to shared/* or
# emulation/* yields no matched subtree -> empty changed_projects -> TheRock
# falls back to building and testing everything. Each entry MUST have a
# corresponding mapping in TheRock (build-topology alias + the test selector's
# _EXTERNAL_SUBTREE_ALIASES); the test selector hard-fails on an unmapped
# shared/* or emulation/* path, so keep this list in lock-step with TheRock when
# adding directories. shared/ctest is intentionally excluded (full-test trigger
# above).
CI_RELEVANT_NON_SUBTREE_PREFIXES = {
    "shared/amdgpu-windows-interop",
    "shared/kpack",
    "shared/machine-readable-isa",
    "emulation/mirage",
    "emulation/rocjitsu",
}


@dataclass
class ConfigureResult:
    changed_projects: str
    run_all_tests: bool
    skip_tests: bool


def get_modified_paths_api(
    github_repo: str, base_sha: str, head_sha: str
) -> Optional[Set[str]]:
    """Return changed paths via `gh api .../compare/base...head`.

    Returns None when GitHub truncates the file list (>= 300 files), signalling
    the caller to fall back to a full run.
    """
    result = subprocess.run(
        ["gh", "api", f"repos/{github_repo}/compare/{base_sha}...{head_sha}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    data = json.loads(result.stdout)
    files = data.get("files", [])
    if len(files) >= 300:
        logger.warning("Compare API returned 300+ files; treating as truncated")
        return None
    return {f["filename"] for f in files}


def is_skippable(path: str) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in SKIPPABLE_PATH_PATTERNS)


def has_non_skippable(paths: Iterable[str]) -> bool:
    return any(not is_skippable(p) for p in paths)


def matches_patterns(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for path in paths for pattern in patterns)


def load_valid_prefixes(config_path: str) -> Set[str]:
    """Read repos-config.json and return the set of ``category/name`` prefixes."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found: {config_path}")
        return set()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load config: {e}")
        return set()
    return {
        f"{e['category']}/{e['name']}"
        for e in data.get("repositories", [])
        if e.get("category") and e.get("name")
    }


def _subtree_prefix(path: str) -> str:
    """First two path components, matching find_matched_subtrees()."""
    return "/".join(path.split("/", 2)[:2])


def find_matched_subtrees(paths: Iterable[str], valid_prefixes: Set[str]) -> List[str]:
    changed_subtrees = {_subtree_prefix(p) for p in paths if len(p.split("/")) >= 2}
    return sorted(changed_subtrees & valid_prefixes)


def get_unclassified_paths(paths: Iterable[str], valid_prefixes: Set[str]) -> List[str]:
    """Non-skippable changed paths that map to no known subtree.

    A PR can mix a recognized path (e.g. ``projects/rdc/...``) with a path we
    cannot classify (a top-level file, or a directory absent from both
    repos-config.json and CI_RELEVANT_NON_SUBTREE_PREFIXES). find_matched_subtrees
    silently drops the unclassified path, which would let CI narrow the build to
    the recognized subset and miss the impact of the unclassified change. Treat
    any such path conservatively (run the full suite).
    """
    return [
        p
        for p in paths
        if not is_skippable(p) and _subtree_prefix(p) not in valid_prefixes
    ]


def configure(
    event_name: str,
    github_repo: str,
    base_sha: Optional[str],
    head_sha: Optional[str],
    config_path: str,
) -> ConfigureResult:
    if event_name in ("schedule", "workflow_dispatch"):
        logger.info(f"{event_name} event - running all tests")
        return ConfigureResult("", True, False)

    if event_name in ("pull_request", "push") and base_sha and head_sha:
        logger.info(f"Getting diff via API: {base_sha}...{head_sha}")
        modified_paths = get_modified_paths_api(github_repo, base_sha, head_sha)
    else:
        logger.warning("No SHAs provided - running all tests")
        return ConfigureResult("", True, False)

    if modified_paths is None:
        logger.info("Truncated API response - running all tests")
        return ConfigureResult("", True, False)

    if not modified_paths:
        logger.info("No modified paths - skipping tests")
        return ConfigureResult("", False, True)

    logger.info(f"Modified paths: {len(modified_paths)} files")

    if matches_patterns(modified_paths, FULL_TEST_TRIGGER_PATTERNS):
        logger.info("CI-infra / full-test-trigger change - running all tests")
        return ConfigureResult("", True, False)

    if not has_non_skippable(modified_paths):
        logger.info("Only skippable files changed - skipping tests")
        return ConfigureResult("", False, True)

    valid_prefixes = load_valid_prefixes(config_path) | CI_RELEVANT_NON_SUBTREE_PREFIXES
    if not valid_prefixes:
        logger.warning("No valid prefixes loaded - running all tests")
        return ConfigureResult("", True, False)

    # Conservative guard: if any non-skippable change cannot be classified to a
    # known subtree, we cannot reason about its impact. Narrowing on the
    # recognized subset would silently drop the unclassified change, so run all.
    unclassified = get_unclassified_paths(modified_paths, valid_prefixes)
    if unclassified:
        logger.info(
            f"Unclassified non-skippable change(s) {sorted(unclassified)[:5]}"
            " - running all tests"
        )
        return ConfigureResult("", True, False)

    matched = find_matched_subtrees(modified_paths, valid_prefixes)
    logger.info(f"Matched projects: {matched}")
    return ConfigureResult(",".join(matched), False, False)


def set_github_output(result: ConfigureResult) -> None:
    outputs = {
        "changed_projects": result.changed_projects,
        "run_all_tests": str(result.run_all_tests).lower(),
        "skip_tests": str(result.skip_tests).lower(),
    }
    logger.info(f"Outputs: {outputs}")
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        for k, v in outputs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure changed projects for rocm-systems multi-arch CI"
    )
    parser.add_argument(
        "--event-name",
        required=True,
        choices=["pull_request", "push", "schedule", "workflow_dispatch"],
    )
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--config-path", default=".github/repos-config.json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger.info(f"Configuring changed projects for {args.github_repo}")
    result = configure(
        event_name=args.event_name,
        github_repo=args.github_repo,
        base_sha=args.base_sha or None,
        head_sha=args.head_sha or None,
        config_path=args.config_path,
    )
    set_github_output(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
