#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices. All rights reserved.
"""Clone Device Metrics Exporter and prepare its submodules.

Replaces the inline bash in ``dme-amdsmi-ci.yml`` Phase 2:

* Clone DME at ``--branch`` into ``--dme-dir``.
* Initialise ``gpuagent`` and ``libamdsmi`` submodules independently so
  that one failing (e.g. private ``libamdsmi``) does not mask the other.
* If ``gpuagent`` did not populate, fall back to a direct clone from the
  GPU Agent repo at ``--gpu-agent-branch``.
* Rewrite SSH URLs in ``gpuagent/.gitmodules`` to HTTPS so nested
  submodules can be initialised in CI environments without SSH keys.
* Fail fast if the protobuf nested submodule is empty -- ``make
  build-libs`` requires it.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from ._common import configure_logging, gh_error, run

logger = logging.getLogger("dme.submodules")

_GIT_SSH_PATTERNS = (re.compile(r"git@github\.com:"), re.compile(r"ssh://git@github\.com/"))
_HTTPS_REPLACEMENT = "https://github.com/"


def _rewrite_gitmodules_ssh_to_https(gitmodules: Path) -> bool:
    """Rewrite SSH GitHub URLs in a ``.gitmodules`` file to HTTPS.

    Returns ``True`` if the file was modified.
    """
    if not gitmodules.is_file():
        return False
    original = gitmodules.read_text()
    rewritten = original
    for pattern in _GIT_SSH_PATTERNS:
        rewritten = pattern.sub(_HTTPS_REPLACEMENT, rewritten)
    if rewritten == original:
        return False
    gitmodules.write_text(rewritten)
    logger.info("Rewrote SSH->HTTPS URLs in %s", gitmodules)
    return True


def _verify_protobuf_submodule(gpuagent_dir: Path) -> None:
    """Hard-fail if the protobuf third-party submodule is empty."""
    proto_dir = gpuagent_dir / "sw/nic/third-party/protobuf"
    if (proto_dir / "autogen.sh").is_file() or (proto_dir / "CMakeLists.txt").is_file():
        return

    gh_error("protobuf submodule is empty -- nested submodule init failed")
    if (gpuagent_dir / ".gitmodules").is_file():
        run(["git", "config", "--file", ".gitmodules", "-l"], cwd=gpuagent_dir, check=False)
    run(["git", "submodule", "status", "--recursive"], cwd=gpuagent_dir, check=False)
    raise SystemExit(1)


def prepare(
    *, dme_repo: str, dme_branch: str, dme_dir: Path, gpu_agent_repo: str, gpu_agent_branch: str
) -> None:
    # CI containers have no SSH keys; force HTTPS for github.com submodule URLs.
    run(["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"])
    # Note: a second --global insteadOf with the same key would overwrite the
    # first, so we only set one and rely on the .gitmodules rewrite below
    # for ``ssh://git@github.com/`` style URLs.

    if dme_dir.exists():
        logger.info("Removing existing DME dir: %s", dme_dir)
        run(["rm", "-rf", str(dme_dir)])

    run(["git", "clone", "-b", dme_branch, dme_repo, str(dme_dir)])

    # Init gpuagent and libamdsmi separately so a failure in one (e.g.
    # private libamdsmi) does not mask the other.
    run(
        ["git", "submodule", "update", "--init", "--recursive", "--", "gpuagent"],
        cwd=dme_dir,
        check=False,
    )
    run(
        ["git", "submodule", "update", "--init", "--recursive", "--", "libamdsmi"],
        cwd=dme_dir,
        check=False,
    )

    gpuagent_dir = dme_dir / "gpuagent"
    if not (gpuagent_dir / "sw").is_dir():
        logger.info("GPU Agent submodule not populated -- cloning %s separately", gpu_agent_repo)
        run(["rm", "-rf", str(gpuagent_dir)])
        run(
            [
                "git",
                "clone",
                "--recurse-submodules",
                "-b",
                gpu_agent_branch,
                gpu_agent_repo,
                str(gpuagent_dir),
            ]
        )

    if _rewrite_gitmodules_ssh_to_https(gpuagent_dir / ".gitmodules"):
        run(["git", "submodule", "sync", "--recursive"], cwd=gpuagent_dir)

    run(["git", "submodule", "update", "--init", "--recursive"], cwd=gpuagent_dir)

    _verify_protobuf_submodule(gpuagent_dir)

    logger.info("DME ready at %s", dme_dir)
    logger.info("GPU Agent ready at %s", gpuagent_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dme-repo", required=True)
    parser.add_argument("--dme-branch", required=True)
    parser.add_argument("--dme-dir", required=True, type=Path)
    parser.add_argument("--gpu-agent-repo", required=True)
    parser.add_argument("--gpu-agent-branch", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    prepare(
        dme_repo=args.dme_repo,
        dme_branch=args.dme_branch,
        dme_dir=args.dme_dir,
        gpu_agent_repo=args.gpu_agent_repo,
        gpu_agent_branch=args.gpu_agent_branch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
