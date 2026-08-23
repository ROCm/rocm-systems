#!/usr/bin/env python3
"""Run pre-commit over a PR's changed files, one config per onboarded subtree.

The repo-root config cannot be used for these subtrees: they are in its
`exclude:` block, so it drops their files before any hook runs and exits 0. For
projects/rccl it is also wrong -- its clang-format v18.1.4 pin cannot parse
projects/rccl/.clang-format ("unknown key 'AlignPPAndNotPP'", exit 1).

Files outside ONBOARDED_SUBTREES are ignored rather than falling back to the
root config, which would apply root hooks to every not-yet-onboarded project on
a merge-gating check.

Arguments:
    --files-from  : File holding the changed paths, one per line (a file, not
                    argv, so paths containing spaces survive).
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

# To onboard a project: add it here and to the `paths:` filter in
# pre-formatting.yml. Its config's hook regexes must be monorepo-relative
# (`^projects/<name>/`) -- pre-commit chdirs to the git toplevel and rewrites
# --files to root-relative paths, so subtree-relative regexes never match.
ONBOARDED_SUBTREES = [
    "projects/rccl",
]

CONFIG_NAME = ".pre-commit-config.yaml"


def config_for(subtree: str) -> str:
    """Return the path to a subtree's own pre-commit config."""
    return f"{subtree}/{CONFIG_NAME}"


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


def read_files_list(path: str) -> List[str]:
    """Read newline-separated paths, dropping blanks and anything not on disk.

    The tree-to-tree diff legitimately names paths the sparse checkout omits.
    pre-commit drops those silently; filtering here keeps the log honest.
    """
    with open(path, encoding="utf-8") as handle:
        raw = [line.rstrip("\n") for line in handle]

    files: List[str] = []
    for entry in raw:
        if not entry.strip():
            continue
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
