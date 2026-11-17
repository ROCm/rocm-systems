# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""isadiff-ci commands, run inside the container (see isadiff-ci on the host)."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_lib as L  # noqa: E402
import comment  # noqa: E402
import scan  # noqa: E402


def init():
    """Clone llvm-project into the llvm volume: from the read-only host clone (fast),
    else from GitHub. Only main is copied; PR heads and branches are fetched on demand.
    """
    repo = L.WORK / "llvm-project"
    if (repo / ".git").exists():
        print(f"{repo} is already initialized")
        return
    src = str(L.HOST_LLVM) if (L.HOST_LLVM / ".git").exists() else L.UPSTREAM
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--no-local",
            "--single-branch",
            "--branch",
            "main",
            src,
            str(repo),
        ],
        check=True,
    )
    L.git(repo, "remote", "remove", "origin")
    scan.fetch_main()
    print(f"{repo} initialized from {src}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="isadiff-ci")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="clone llvm-project into the llvm volume (once)")
    s = sub.add_parser("scan", help="run rj_isa_diff_pr on new PR and branch heads")
    s.add_argument("--tier", choices=("quick", "deep"), default="quick")
    s.add_argument("--pr", type=int, help="only this PR, of any author")
    p = sub.add_parser(
        "post", help="print (or with --yes post) the latest run's PR comment"
    )
    p.add_argument("pr", type=int)
    p.add_argument("--yes", action="store_true")
    r = sub.add_parser(
        "reply", help="reply to a review comment; the body is read from stdin"
    )
    r.add_argument("pr", type=int)
    r.add_argument("comment_id", type=int)
    r.add_argument("--yes", action="store_true")
    sub.add_parser("index", help="rewrite results/index.md")
    a = ap.parse_args(argv)
    cfg = L.load_config()
    if a.cmd == "init":
        init()
    elif a.cmd == "scan":
        scan.scan(cfg, a.tier, a.pr)
    elif a.cmd == "post":
        comment.post(cfg, a.pr, a.yes)
    elif a.cmd == "reply":
        comment.reply(cfg, a.pr, a.comment_id, sys.stdin.read(), a.yes)
    elif a.cmd == "index":
        scan.write_index()


if __name__ == "__main__":
    main()
