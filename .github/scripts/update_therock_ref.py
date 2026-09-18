#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bumps the pinned ROCm/TheRock commit across config and wrapper workflows.

Updates .github/therock_ref.json plus the two literal SHA occurrences
(job `uses:@sha` and `with: ref: sha`) in each of the three wrapper
reusable workflows under .github/workflows/_therock_*.yml. These files
are the only remaining hardcoded TheRock commit pins in this repo; every
other workflow reads the ref dynamically via .github/actions/therock-ref.

Usage:
    python .github/scripts/update_therock_ref.py <new_sha> [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / ".github" / "therock_ref.json"
WRAPPER_PATHS = [
    REPO_ROOT / ".github" / "workflows" / "_therock_setup_multi_arch.yml",
    REPO_ROOT / ".github" / "workflows" / "_therock_multi_arch_ci_linux.yml",
    REPO_ROOT / ".github" / "workflows" / "_therock_multi_arch_ci_windows.yml",
]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
THEROCK_URL = "https://github.com/ROCm/TheRock.git"


def fetch_commit_date(sha: str) -> str:
    """Returns sha's committer timestamp (UTC, e.g. 2026-09-18T07:13:59Z) via git."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(
            ["git", "fetch", "-q", "--depth=1", THEROCK_URL, sha],
            check=True,
            cwd=tmp,
        )
        result = subprocess.run(
            ["git", "show", "-s", "--date=format-local:%Y-%m-%dT%H:%M:%SZ", "--format=%cd", "FETCH_HEAD"],
            check=True,
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**os.environ, "TZ": "UTC"},
        )
    return result.stdout.strip()


def substitute_sha(text: str, old_sha: str, new_sha: str, new_date: str) -> tuple[str, int]:
    pattern = re.compile(rf"{re.escape(old_sha)}( # \S+)?")
    new_text, count = pattern.subn(f"{new_sha} # {new_date}", text)
    return new_text, count


def update_config(config: dict, new_sha: str, new_commit_date: str) -> dict:
    return {
        "ref": new_sha,
        "commit_date": new_commit_date,
        "updated_date": date.today().isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_sha", help="New ROCm/TheRock commit SHA (40 hex chars)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without writing files."
    )
    args = parser.parse_args()

    new_sha = args.new_sha.strip().lower()
    if not SHA_RE.match(new_sha):
        parser.error(f"new_sha must be 40 hex characters, got: {args.new_sha!r}")

    new_commit_date = fetch_commit_date(new_sha)

    old_config = json.loads(CONFIG_PATH.read_text())
    old_sha = old_config["ref"]
    new_config = update_config(old_config, new_sha, new_commit_date)

    print(f"Bumping TheRock ref: {old_sha} -> {new_sha} ({new_commit_date})")

    if not args.dry_run:
        CONFIG_PATH.write_text(json.dumps(new_config, indent=2) + "\n")
    print(f"  {'[dry-run] ' if args.dry_run else ''}wrote {CONFIG_PATH}")

    for wrapper_path in WRAPPER_PATHS:
        text = wrapper_path.read_text()
        new_text, count = substitute_sha(text, old_sha, new_sha, new_commit_date)
        if count != 2:
            raise SystemExit(
                f"Expected exactly 2 occurrences of {old_sha} in {wrapper_path}, "
                f"found {count}. The wrapper file may have drifted from the "
                "expected shape; fix it before re-running this script."
            )
        if not args.dry_run:
            wrapper_path.write_text(new_text)
        print(
            f"  {'[dry-run] ' if args.dry_run else ''}updated {count} occurrence(s) "
            f"in {wrapper_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
