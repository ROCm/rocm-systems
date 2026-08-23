#!/usr/bin/env python3
"""Run pre-commit over a PR's changed files, one config per onboarded subtree.

The monorepo has no single pre-commit config. Several subtrees carry their own
`.pre-commit-config.yaml` and are listed in the repo-root config's `exclude:`
block, which means running the root config over their files silently checks
nothing (pre-commit applies the top-level `exclude:` before any hook runs, then
exits 0). Worse, for projects/rccl the root config is actively wrong: it pins
clang-format v18.1.4, which cannot even parse projects/rccl/.clang-format and
exits 1 with "unknown key 'AlignPPAndNotPP'".

This script therefore groups the changed files by subtree and invokes
pre-commit once per group with that subtree's own config.

Files that do not belong to an ONBOARDED_SUBTREES entry are IGNORED, not
checked against the root config. That is deliberate: a root fallback would
apply root hooks (black, gersemi, clang-format 18) to every not-yet-onboarded
project the moment this goes live, on a merge-gating check. Onboarding a
project is a two-line change -- add it here, and add its path to the `paths:`
filter in .github/workflows/pre-formatting.yml.

Note: paths are always matched monorepo-root-relative. pre-commit chdirs to the
git toplevel and rewrites both --config and every --files argument before doing
anything, so a subtree config's `files:`/`exclude:` regexes must be written with
the full `^projects/<name>/` prefix regardless of how it is invoked.

Arguments:
    --files-from  : Path to a file holding the changed paths, one per line.
                    A file is used rather than argv so that paths containing
                    spaces survive intact.
    --subtree     : OPTIONAL, repeatable. Override ONBOARDED_SUBTREES (testing).
    --dry-run     : If set, log the pre-commit invocations without running them.
    --debug       : If set, enables detailed debug logging.

Outputs:
    Exit 0 if every group passed (or nothing onboarded was touched), 1 if any
    group failed or could not be run.

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

# Subtrees whose formatting is enforced. Add one line per project as it
# onboards, and mirror it in the `paths:` filter of pre-formatting.yml.
#
# Each entry must contain a .pre-commit-config.yaml whose hook `files:` regexes
# are written monorepo-relative (see the module docstring).
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

    Files matching no onboarded subtree are omitted entirely. Longest prefix
    wins, so a nested subtree takes precedence over its parent.
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

    The sparse checkout only materialises onboarded subtrees plus .github, so a
    tree-to-tree diff legitimately names paths that are absent. pre-commit drops
    those silently; filtering here keeps the logs honest about what was checked.
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
        # Fail loudly. Falling back to the root config here would silently
        # check nothing, because onboarded subtrees are in its `exclude:`.
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
