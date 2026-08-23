#!/usr/bin/env python3
"""Run pre-commit over a PR's changed files, one config per onboarded subtree.

The repo-root config is in `exclude:` for these subtrees, so it checks nothing and exits 0; for
rccl its clang-format v18.1.4 also cannot parse projects/rccl/.clang-format.

Arguments:
    --files-from  : File holding the changed paths, one per line.
    --subtree     : OPTIONAL, repeatable. Override ONBOARDED_SUBTREES (testing).
    --dry-run     : If set, log the pre-commit invocations without running them.
    --debug       : If set, enables detailed debug logging.

Outputs:
    Exit 0 if every group passed or nothing onboarded was touched, else 1.

Example Usage:
    python .github/scripts/run_pre_commit_by_subtree.py --files-from changed.txt --debug
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Onboarding needs all four: here; `paths:` in pre-formatting.yml; a category/name entry in
# .github/repos-config.json (the only source the checkout can materialise from); and the root
# .pre-commit-config.yaml `exclude:`. Hook regexes must be monorepo-relative (`^projects/<name>/`).
ONBOARDED_SUBTREES = [
    "projects/rccl",
]

CONFIG_NAME = ".pre-commit-config.yaml"


def config_for(subtree: str) -> str:
    """Return the path to a subtree's own pre-commit config."""
    return f"{subtree}/{CONFIG_NAME}"


def validate_subtrees(subtrees: List[str]) -> None:
    """Reject spellings that match nothing: it is a literal prefix compare, so a trailing slash or
    a "./" prefix owns no files and silently greens the gate."""
    for subtree in subtrees:
        if not subtree or subtree != subtree.strip():
            raise ValueError(f"onboarded subtree {subtree!r} is empty or padded")
        if subtree.endswith("/") or subtree.startswith("/"):
            raise ValueError(
                f"onboarded subtree {subtree!r} must have no leading or "
                "trailing slash; it is compared as a literal path prefix"
            )
        if subtree.startswith("./") or ".." in subtree.split("/"):
            raise ValueError(
                f"onboarded subtree {subtree!r} must be a plain repo-relative "
                "path such as 'projects/rccl'"
            )


def group_files_by_subtree(
    files: List[str], subtrees: List[str]
) -> Dict[str, List[str]]:
    """Group changed files by the onboarded subtree that owns them.

    Unowned files are omitted. Longest prefix wins.
    """
    groups: Dict[str, List[str]] = {}
    ordered = sorted(subtrees, key=len, reverse=True)
    for path in files:
        for subtree in ordered:
            if path == subtree or path.startswith(subtree + "/"):
                groups.setdefault(subtree, []).append(path)
                break
        else:
            logger.debug("ignoring %s (no onboarded subtree owns it)", path)
    return groups


def check_subtrees_materialised(raw_files: List[str], subtrees: List[str]) -> List[str]:
    """Return onboarded subtrees that own changed files but were never checked out -- otherwise
    every path is dropped as absent and the job exits 0 having checked nothing.

    Keyed off the subtree's config, not its changed files, so a delete-only PR is not a false red.
    """
    missing: List[str] = []
    for subtree in sorted(group_files_by_subtree(raw_files, subtrees)):
        if not os.path.isfile(config_for(subtree)):
            missing.append(subtree)
    return missing


def read_paths(path: str) -> List[str]:
    """Read newline-separated paths verbatim, dropping only blank lines."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def read_files_list(path: str) -> List[str]:
    """Read paths, dropping blanks and anything not on disk (the tree diff names files the sparse
    checkout omits). Callers MUST run check_subtrees_materialised first -- this filter hides an
    under-reporting `detect` step."""
    files: List[str] = []
    for entry in read_paths(path):
        if not os.path.isfile(entry):
            logger.debug("skipping %s (not present in sparse checkout)", entry)
            continue
        files.append(entry)
    return files


def run_group(subtree: str, files: List[str], dry_run: bool) -> bool:
    """Run pre-commit for one subtree. Return True on success."""
    config = config_for(subtree)
    if not os.path.isfile(config):
        logger.error(
            "%s is onboarded but %s does not exist; refusing to fall back "
            "to the root config, which would pass without checking anything.",
            subtree,
            config,
        )
        return False

    cmd = [
        "pre-commit",
        "run",
        "-c",
        config,
        "--show-diff-on-failure",
        "--color=always",
        "--files",
        *files,
    ]
    logger.info("[%s] %d file(s) via %s", subtree, len(files), config)
    for path in files:
        logger.info("    %s", path)

    if dry_run:
        logger.info("[%s] dry-run, not executing", subtree)
        return True

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        logger.error("[%s] pre-commit failed (exit %d)", subtree, completed.returncode)
        return False

    logger.info("[%s] passed", subtree)
    return True


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pre-commit per onboarded subtree, using that subtree's config."
    )
    parser.add_argument(
        "--files-from",
        required=True,
        help="File containing the changed paths, one per line",
    )
    parser.add_argument(
        "--subtree",
        action="append",
        default=None,
        help="Override the onboarded subtree list (repeatable; for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the pre-commit invocations without running them",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Group the PR's changed files by subtree and run pre-commit on each."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    subtrees = args.subtree if args.subtree else ONBOARDED_SUBTREES
    validate_subtrees(subtrees)

    raw_files = read_paths(args.files_from)
    unmaterialised = check_subtrees_materialised(raw_files, subtrees)
    if unmaterialised:
        for subtree in unmaterialised:
            logger.error(
                "%s has changed files but %s is not on disk, so it was never checked out. "
                "Confirm it is a category/name entry in .github/repos-config.json.",
                subtree,
                config_for(subtree),
            )
        sys.exit(1)

    files = read_files_list(args.files_from)
    groups = group_files_by_subtree(files, subtrees)

    if not groups:
        logger.info(
            "No files under an onboarded subtree (%s); nothing to check.",
            ", ".join(subtrees),
        )
        return

    failed: List[str] = []
    for subtree in sorted(groups):
        if not run_group(subtree, groups[subtree], args.dry_run):
            failed.append(subtree)

    if failed:
        logger.error("pre-commit failed for: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("All %d subtree group(s) passed.", len(groups))


if __name__ == "__main__":
    main()
