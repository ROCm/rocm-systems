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
    Exit 1 without running any hook if an onboarded subtree has changed files
    but was never checked out -- see check_subtrees_materialised.

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

# To onboard a project, ALL FOUR of these are required:
#
#   1. Add its path here, as `category/name` with no trailing slash.
#   2. Add its path to the `paths:` filter in pre-formatting.yml, or the gate
#      never runs on its PRs.
#   3. Make sure it is a `category/name` entry in .github/repos-config.json.
#      That file is the ONLY thing pr_detect_changed_subtrees.py can emit, and
#      its output is what puts the subtree's files on disk via the workflow's
#      sparse-checkout. Every entry there is currently `category: projects`, so
#      a subtree under emulation/, shared/ or python/ needs a repos-config
#      entry adding first. Skipping this used to produce a silent green;
#      check_subtrees_materialised below now makes it a hard failure.
#   4. Add it to the repo-root .pre-commit-config.yaml `exclude:` block, so the
#      root config stops claiming files that this script checks with a
#      different one.
#
# Its config's hook regexes must be monorepo-relative (`^projects/<name>/`) --
# pre-commit chdirs to the git toplevel and rewrites --files to root-relative
# paths, so subtree-relative regexes never match.
ONBOARDED_SUBTREES = [
    "projects/rccl",
]

CONFIG_NAME = ".pre-commit-config.yaml"


def config_for(subtree: str) -> str:
    """Return the path to a subtree's own pre-commit config."""
    return f"{subtree}/{CONFIG_NAME}"


def validate_subtrees(subtrees: List[str]) -> None:
    """Reject subtree spellings that would silently match nothing.

    group_files_by_subtree compares against `subtree` and `subtree + "/"`, so
    "projects/rccl/" or "./projects/rccl" own no files at all -- yielding an
    empty group set, "nothing to check", and a permanently green gate with no
    diagnostic. Fail at startup instead.
    """
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


def check_subtrees_materialised(raw_files: List[str], subtrees: List[str]) -> List[str]:
    """Return onboarded subtrees that own changed files but are not on disk.

    The gate's file list is a git tree diff, which sparse-checkout does not
    filter; the files themselves arrive only if pr_detect_changed_subtrees.py
    named the subtree. When it does not -- the subtree is missing from
    repos-config.json, or get_changed_files returned a partial page and the
    caller could not tell -- every path is dropped as "not present in sparse
    checkout" and the job exits 0 having checked nothing.

    A subtree's own config is the cheapest proof that its cone was checked
    out. Using it rather than the changed files themselves means a PR that
    only DELETES files here is still fine: the config is present, the files
    legitimately are not.
    """
    missing = []
    for subtree in sorted(group_files_by_subtree(raw_files, subtrees)):
        if not os.path.isfile(config_for(subtree)):
            missing.append(subtree)
    return missing


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


def read_paths(path: str) -> List[str]:
    """Read newline-separated paths verbatim, dropping only blank lines."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def read_files_list(path: str) -> List[str]:
    """Read newline-separated paths, dropping blanks and anything not on disk.

    The tree-to-tree diff legitimately names paths the sparse checkout omits.
    pre-commit drops those silently; filtering here keeps the log honest.

    This filter is also the only thing between an under-reporting `detect`
    step and a green gate, so callers MUST run check_subtrees_materialised
    over the unfiltered list before trusting a small result.
    """
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
                "%s has changed files but %s is not on disk: its sparse "
                "checkout never happened. Check that it is a category/name "
                "entry in .github/repos-config.json and that the detect step "
                "listed it. Refusing to report a green check for files that "
                "were never looked at.",
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
