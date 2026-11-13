#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regenerate the vendored ELFIO headers from upstream plus local patches.

The upstream repository and commit come from ``SOURCE`` (``<url> <commit>``).
The script fetches that commit, applies ``patches/*.patch`` in name order, and
replaces ``elfio/`` and ``LICENSE.txt`` next to this script with the result.

Usage:
    update.py            regenerate the vendored files
    update.py --check    verify the vendored files match SOURCE + patches
    update.py --commit <ref>
                         move to another upstream commit or ref and record
                         its full SHA in SOURCE. Delete patches that the new
                         commit already contains first, or they fail to apply.

Every vendored file must be reproducible this way, so never edit ``elfio/``
by hand: add or change a patch and rerun the script.
"""

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = SCRIPT_DIR / "SOURCE"
PATCH_DIR = SCRIPT_DIR / "patches"
HEADER_DIR = "elfio"
LICENSE = "LICENSE.txt"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def read_source() -> tuple[str, str]:
    url, commit = SOURCE_FILE.read_text().split()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        sys.exit(f"{SOURCE_FILE.name}: pin must be a full commit SHA, not {commit!r}")
    return url, commit


def build_tree(url: str, commit: str, work: Path) -> Path:
    """Fetch upstream at ``commit`` into ``work`` and apply the patches."""
    repo = work / "ELFIO"
    repo.mkdir()
    git("init", "--quiet", cwd=repo)
    # Check out upstream's bytes regardless of the user's Git configuration.
    git("config", "core.autocrlf", "false", cwd=repo)
    git("config", "core.eol", "lf", cwd=repo)
    git("fetch", "--quiet", "--depth", "1", url, commit, cwd=repo)
    git(
        "-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD", cwd=repo
    )
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        print(f"applying {patch.name}")
        try:
            git("apply", "--whitespace=nowarn", str(patch), cwd=repo)
        except subprocess.CalledProcessError:
            sys.exit(
                f"{patch.name} does not apply to {commit}. If upstream already "
                "contains it, delete the patch and rerun."
            )
    return repo


def resolved_commit(repo: Path) -> str:
    """Full SHA of the checked-out upstream commit."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def differences(repo: Path) -> list[str]:
    """List vendored files that differ from, or are missing in, ``repo``."""
    diffs = []
    for name in (HEADER_DIR, LICENSE):
        ours, theirs = SCRIPT_DIR / name, repo / name
        if theirs.is_file():
            if not ours.is_file() or not filecmp.cmp(ours, theirs, shallow=False):
                diffs.append(name)
            continue
        ours_files = {p.name for p in ours.glob("*")} if ours.is_dir() else set()
        theirs_files = {p.name for p in theirs.glob("*")}
        for fname in sorted(ours_files | theirs_files):
            a, b = ours / fname, theirs / fname
            if (
                not a.is_file()
                or not b.is_file()
                or not filecmp.cmp(a, b, shallow=False)
            ):
                diffs.append(f"{name}/{fname}")
    return diffs


def install(repo: Path) -> None:
    shutil.rmtree(SCRIPT_DIR / HEADER_DIR, ignore_errors=True)
    shutil.copytree(repo / HEADER_DIR, SCRIPT_DIR / HEADER_DIR)
    shutil.copyfile(repo / LICENSE, SCRIPT_DIR / LICENSE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="verify the vendored files instead of rewriting them",
    )
    group.add_argument(
        "--commit",
        help="upstream commit or ref to move SOURCE to (recorded as a full SHA)",
    )
    args = parser.parse_args()

    url, commit = read_source()
    if args.commit:
        commit = args.commit

    with tempfile.TemporaryDirectory() as tmp:
        repo = build_tree(url, commit, Path(tmp))
        if args.check:
            diffs = differences(repo)
            if diffs:
                print("vendored ELFIO does not match SOURCE + patches:")
                for d in diffs:
                    print(f"  {d}")
                return 1
            print("vendored ELFIO matches SOURCE + patches")
            return 0
        install(repo)
        commit = resolved_commit(repo)

    if args.commit:
        SOURCE_FILE.write_text(f"{url} {commit}\n")
    print(f"vendored ELFIO updated to {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
