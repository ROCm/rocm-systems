# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check the HRR sources against their base and the Catch2 sources.

  api-compat  Check that the checked-in API IDs stay compatible with the
              header on the base commit (IDs are part of the archive format).
  workloads   Read the API matrix tier workload lists against the Catch2
              sources.
  all         Both, reporting every failure.

That the checked-in generated files are current is tools/tests/
test_generated_files_current.py, which the script tests job runs.
Everything is written under --out; checked-in files are never modified.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

Runner = Callable[[Sequence[str]], int]

COMPAT_CHECKER = "projects/hrr/tools/check_hrr_api_compat.py"
MATRIX_TOOLS = "projects/hrr/tools/api-matrix"
API_ARGS_HEADER = "projects/hrr/include/hrr/hrr_api_args.h"


def run(cmd: Sequence[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(list(cmd)).returncode


def error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def base_header(root: Path, base_sha: str) -> tuple[str | None, str | None]:
    """hrr_api_args.h at base_sha.

    Returns (content, None) on success, (None, None) when the path genuinely
    does not exist at base_sha -- the only case with nothing to compare --
    and (None, reason) for any other git failure. The base commit must be
    verified to resolve first: in a plain checkout an unresolvable rev and
    a missing path print the same 'does not exist in' text, and the
    codegen job's ``blob:none`` clone can also report 'not our ref' for a
    commit it does not have -- an unreadable base must fail the check
    instead of passing an incompatible API-ID renumbering.
    """
    verify = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet",
         f"{base_sha}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        detail = (verify.stderr or verify.stdout).strip()
        return None, (
            f"cannot read {API_ARGS_HEADER} at {base_sha}: the base commit does not "
            f"resolve ({detail or f'git rev-parse exited {verify.returncode}'})"
        )
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{base_sha}:{API_ARGS_HEADER}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc.stdout, None
    detail = (proc.stderr or proc.stdout).strip()
    if re.search(r"does not exist in|exists on disk, but not in", detail):
        return None, None
    if re.search(r"not our ref|error:Server does not allow request", detail):
        return None, (
            f"cannot read {API_ARGS_HEADER} at {base_sha}: "
            f"the blob could not be fetched ({detail.splitlines()[-1]})"
        )
    return None, (
        f"cannot read {API_ARGS_HEADER} at {base_sha}: "
        f"{detail or f'git show exited {proc.returncode}'}"
    )


def resolve_base(root: Path, base_sha: str) -> str:
    """The commit to compare API IDs against: base_sha, else the parent commit."""
    if base_sha and set(base_sha) != {"0"}:
        return base_sha
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def check_api_compat(
    root: Path,
    out: Path,
    base_sha: str,
    runner: Runner = run,
    read_base: Callable[[Path, str], tuple[str | None, str | None]] = base_header,
) -> bool:
    current = root / API_ARGS_HEADER
    text, failure = read_base(root, base_sha)
    if failure:
        error(failure)
        error("the API-ID compatibility with the base could not be verified; do not pass it silently")
        return False
    if text is None:
        print(f"{API_ARGS_HEADER} does not exist at {base_sha}; nothing to compare")
        return True
    base = out / "base_hrr_api_args.h"
    base.write_text(text, encoding="utf-8")
    rc = runner(
        [
            sys.executable, str(root / COMPAT_CHECKER),
            "--base-header", str(base),
            "--current-header", str(current),
        ]
    )
    if rc != 0:
        error(f"HRR API IDs are not compatible with {base_sha}; see {COMPAT_CHECKER}")
        return False
    return True


def check_workloads(root: Path, runner: Runner = run) -> bool:
    rc = runner(
        [sys.executable, str(root / MATRIX_TOOLS / "check_matrix.py"), "--check-workloads"]
    )
    if rc != 0:
        error(f"check_matrix.py --check-workloads failed (exit {rc})")
    return rc == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("api-compat", "workloads", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("build/hrr-codegen"))
    parser.add_argument("--base-sha", default="")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ok = True
    if args.check in ("api-compat", "all"):
        ok &= check_api_compat(root, out, resolve_base(root, args.base_sha))
    if args.check in ("workloads", "all"):
        ok &= check_workloads(root)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
