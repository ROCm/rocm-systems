#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Clone dynolog at a pinned ref for the RDC integration build.

Only the in-tree RDC shim (``dynolog/src/gpumon/amd``) is needed, so this does
a shallow, single-commit fetch and skips submodules. The ref may be a branch,
tag, or full 40-char commit SHA; GitHub allows fetching an arbitrary SHA
directly, which keeps a pinned automatic run reproducible without cloning the
whole (large) dynolog history.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ._common import configure_logging, gh_error, run

logger = logging.getLogger("dynolog.clone")

# Force SSH->HTTPS on CI hosts without SSH keys, without mutating the runner's
# persistent global git config (per-command ``-c`` entries only). Only the
# fetch resolves the remote URL, so only it needs these.
_INSTEAD_OF = [
    "-c",
    "url.https://github.com/.insteadOf=git@github.com:",
    "-c",
    "url.https://github.com/.insteadOf=ssh://git@github.com/",
]

# The RDC shim source lives here; a successful clone must contain it.
_RDC_SHIM_SUBPATH = Path("dynolog/src/gpumon/amd/RdcWrapper.cpp")


def clone(*, repo: str, ref: str, dest: Path) -> Path:
    """Shallow-clone ``repo`` at ``ref`` into ``dest`` and return ``dest``."""
    if ref.startswith("-"):
        gh_error(f"refusing suspicious ref that looks like an option: {ref!r}")
        raise SystemExit(1)
    if dest.exists():
        logger.info("Removing existing dynolog dir: %s", dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    run(["git", "init", "--quiet", str(dest)])
    run(["git", "remote", "add", "origin", repo], cwd=dest)

    # ``git fetch --depth 1 <ref>`` accepts a branch, tag, or (on GitHub) a full
    # commit SHA, fetching just that one commit instead of the whole history.
    run(["git", *_INSTEAD_OF, "fetch", "--depth", "1", "origin", ref], cwd=dest)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)

    shim = dest / _RDC_SHIM_SUBPATH
    if not shim.is_file():
        gh_error(f"dynolog clone missing RDC shim: {shim} (bad ref '{ref}'?)")
        raise SystemExit(1)

    logger.info("dynolog ready at %s (ref=%s)", dest, ref)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="https://github.com/facebookincubator/dynolog.git")
    parser.add_argument(
        "--ref", required=True, help="Branch, tag, or full 40-char commit SHA to check out."
    )
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    clone(repo=args.repo, ref=args.ref, dest=args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
