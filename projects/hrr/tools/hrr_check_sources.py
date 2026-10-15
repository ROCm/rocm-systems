# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check the HRR sources that are generated from the HIP API headers.

  generated   Regenerate hrr_api_args.h and the capture and playback dispatch
              sources and compare them with the checked-in copies.
  api-compat  Check that the regenerated API IDs stay compatible with the
              header on the base commit (IDs are part of the archive format).
  api-matrix  Re-derive the API class manifest and compare the generated
              test expectations with the checked-in header.
  all         All three, reporting every failure.

Everything is written under --out; checked-in files are never modified.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

Runner = Callable[[Sequence[str]], int]

API_TRACE = "projects/clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp"
PUBLIC_HEADER = "projects/hip/include/hip/hip_runtime_api.h"
MANUAL_CAPTURE = "projects/clr/hipamd/src/hrr/hip_capture.cpp"
GENERATOR = "projects/hrr/tools/gen_hrr_api_args.py"
COMPAT_CHECKER = "projects/hrr/tools/check_hrr_api_compat.py"
MATRIX_TOOLS = "projects/hrr/tools/api-matrix"
API_ARGS_HEADER = "projects/hrr/include/hrr/hrr_api_args.h"
MATRIX_EXPECTATIONS = "projects/hrr/tests/integration/hrr_api_matrix_expectations.h"

# Checked-in file -> name of the regenerated copy under --out.
GENERATED_FILES = {
    API_ARGS_HEADER: "include/hrr/hrr_api_args.h",
    "projects/clr/hipamd/src/hrr/hip_capture_generated.cpp": "hip_capture_generated.cpp",
    "projects/hrr/playback/hip_playback_generated.cpp": "hip_playback_generated.cpp",
}


def run(cmd: Sequence[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(list(cmd)).returncode


def error(message: str) -> None:
    print(f"::error::{message}", flush=True)


def diff_files(expected: Path, actual: Path) -> list[str]:
    """Unified diff of two text files, ignoring line-ending differences."""
    a = expected.read_text(encoding="utf-8").splitlines()
    b = actual.read_text(encoding="utf-8").splitlines()
    return list(
        difflib.unified_diff(a, b, fromfile=str(expected), tofile=str(actual), lineterm="")
    )


def check_generated(root: Path, out: Path, runner: Runner = run) -> bool:
    (out / "include" / "hrr").mkdir(parents=True, exist_ok=True)
    rc = runner(
        [
            sys.executable, str(root / GENERATOR),
            "--input", str(root / API_TRACE),
            "--public-header", str(root / PUBLIC_HEADER),
            "--output-header", str(out / GENERATED_FILES[API_ARGS_HEADER]),
            "--output-capture", str(out / "hip_capture_generated.cpp"),
            "--output-playback", str(out / "hip_playback_generated.cpp"),
            "--check-hrr-coverage",
            "--manual-capture-source", str(root / MANUAL_CAPTURE),
            "--silent",
        ]
    )
    if rc != 0:
        error(f"{GENERATOR} failed (exit {rc})")
        return False

    stale = []
    for tracked, generated in GENERATED_FILES.items():
        lines = diff_files(root / tracked, out / generated)
        if lines:
            print("\n".join(lines))
            stale.append(tracked)
    if stale:
        error(
            "checked-in HRR generated sources are stale: "
            + ", ".join(stale)
            + f". Regenerate them with {GENERATOR} --check-hrr-coverage."
        )
        return False
    return True


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
    current = out / GENERATED_FILES[API_ARGS_HEADER]
    if not current.is_file():
        error("api-compat needs the regenerated header; run the generated check first")
        return False
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


def check_api_matrix(root: Path, out: Path, runner: Runner = run) -> bool:
    tools = root / MATRIX_TOOLS
    matrix_out = out / "api-matrix"
    matrix_out.mkdir(parents=True, exist_ok=True)
    manifest = matrix_out / "api_classes.json"
    emitted = matrix_out / "hrr_api_matrix_expectations.h"
    steps = [
        [sys.executable, str(tools / "derive_manifest.py"), "--quiet", "--output", str(manifest)],
        [sys.executable, str(tools / "check_matrix.py"), "--check-workloads"],
        [
            sys.executable, str(tools / "check_matrix.py"),
            "--manifest", str(manifest), "--emit-cxx", str(emitted),
        ],
    ]
    for step in steps:
        rc = runner(step)
        if rc != 0:
            error(f"{Path(step[1]).name} failed (exit {rc})")
            return False
    lines = diff_files(root / MATRIX_EXPECTATIONS, emitted)
    if lines:
        print("\n".join(lines))
        error(
            f"{MATRIX_EXPECTATIONS} is stale; run derive_manifest.py, then "
            f"check_matrix.py --emit-cxx, in {MATRIX_TOOLS}"
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("check", choices=("generated", "api-compat", "api-matrix", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("build/hrr-codegen"))
    parser.add_argument("--base-sha", default="")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ok = True
    if args.check in ("generated", "all"):
        ok &= check_generated(root, out)
    if args.check in ("api-compat", "all"):
        ok &= check_api_compat(root, out, resolve_base(root, args.base_sha))
    if args.check in ("api-matrix", "all"):
        ok &= check_api_matrix(root, out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
