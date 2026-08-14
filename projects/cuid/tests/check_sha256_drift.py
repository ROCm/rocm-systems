#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Fail if the two copies of SHA-256 in this repo have drifted apart.

libamdcuid carries a copy of rocprofiler-sdk's SHA-256. A silent divergence
would not fail to build -- it would change every CUID ever issued.

    python3 projects/cuid/tests/check_sha256_drift.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Both paths are relative to the repository root.
UPSTREAM = Path("projects/rocprofiler-sdk/source/lib/common/sha256.cpp")
COPY = Path("projects/cuid/lib/src/sha256.cc")

# update() may differ only by rocprofiler's logging call. Anything else is drift.
ALLOWED_UPSTREAM_ONLY = ("ROCP_CI_LOG_IF",)


def strip_comments(src: str) -> str:
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)


def member_bodies(path: Path) -> dict[tuple[str, str], str]:
    """Map (name, signature) -> body text, for every member defined in the file.

    The signature includes the parameter list and any trailing cv/ref
    qualifiers, so overloads (e.g. two sha256::update()) get distinct keys
    instead of colliding on the member name alone.
    """
    src = strip_comments(path.read_text(encoding="utf-8"))
    bodies: dict[tuple[str, str], str] = {}
    for match in re.finditer(r"\bsha256::(\w+)\s*\(", src):
        name = match.group(1)

        depth, idx = 1, match.end()
        while idx < len(src) and depth > 0:
            if src[idx] == "(":
                depth += 1
            elif src[idx] == ")":
                depth -= 1
            idx += 1
        params_end = idx

        open_brace = src.find("{", params_end)
        if open_brace < 0:
            continue
        signature = re.sub(r"\s+", "", src[match.end() - 1 : open_brace])

        depth, idx = 0, open_brace
        while idx < len(src):
            if src[idx] == "{":
                depth += 1
            elif src[idx] == "}":
                depth -= 1
                if depth == 0:
                    break
            idx += 1
        bodies.setdefault((name, signature), src[open_brace + 1 : idx])
    return bodies


def normalise(body: str, drop_allowed: bool) -> str:
    lines = []
    for line in body.split("\n"):
        if drop_allowed and any(tok in line for tok in ALLOWED_UPSTREAM_ONLY):
            continue
        lines.append(line)
    return re.sub(r"\s+", "", "\n".join(lines))


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    upstream_path, copy_path = root / UPSTREAM, root / COPY

    for path in (upstream_path, copy_path):
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    upstream = member_bodies(upstream_path)
    copy = member_bodies(copy_path)

    if not upstream or not copy:
        print(
            "error: parsed no sha256:: members; the extractor needs updating",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []

    missing = sorted(set(upstream) - set(copy))
    if missing:
        names = ", ".join(f"{name}{sig}" for name, sig in missing)
        problems.append(f"{COPY} is missing member(s) present upstream: {names}")

    for key in sorted(set(upstream) & set(copy)):
        if normalise(upstream[key], drop_allowed=True) != normalise(copy[key], False):
            name, sig = key
            problems.append(f"sha256::{name}{sig} differs between the two copies")

    if problems:
        print("SHA-256 implementations have drifted apart:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            f"\n  upstream: {UPSTREAM}\n  copy:     {COPY}\n\n"
            "Reconcile them, or record an intended difference in ALLOWED_UPSTREAM_ONLY.",
            file=sys.stderr,
        )
        return 1

    print(
        f"sha256 copies agree: {len(set(upstream) & set(copy))} member functions checked"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
