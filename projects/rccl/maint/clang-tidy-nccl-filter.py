#!/usr/bin/env python3
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Filter clang-tidy diagnostics down to code owned by the RCCL team, dropping
# any finding that lands on a line still owned entirely by upstream NCCL.
#
# Why this exists
# ---------------
# RCCL is a fork of NVIDIA NCCL maintained as a subtree merge. The vast
# majority of the host sources come straight from upstream; complaining about
# the complexity of NCCL-owned functions is noise the RCCL team cannot act on
# (the fix belongs upstream, and any local change is undone at the next sync).
# This filter keeps clang-tidy focused on the lines RCCL has actually touched
# since the last NCCL sync.
#
# How ownership is decided
# ------------------------
# The last NCCL sync is a merge commit whose SECOND parent is the NCCL-owned
# side (e.g. "merge(rccl): sync NCCL vX.Y.Z into projects/rccl"). For each host
# source file, its content is diffed (difflib) against the corresponding
# upstream NCCL file at that sync commit. A line is "NCCL-owned" iff it falls
# in an unchanged ("equal") block of that diff, i.e. its content is identical
# to upstream. Lines RCCL added or changed relative to upstream, and files with
# no upstream counterpart at all, are RCCL-owned and kept. This is a pure
# content comparison, so it is fast and does not depend on git-blame history
# rewriting done by the sync tooling.
#
# Known limitation: ownership is decided per line at the diagnostic's location.
# A complexity finding is reported on a function's signature line; if RCCL only
# edited the body of an otherwise-upstream function, the signature line may
# still match upstream and the finding will be dropped.
#
# hipify line mapping
# -------------------
# clang-tidy analyses the hipified copies under build/<type>/hipify/src, whose
# line numbers differ slightly from the original projects/rccl/src sources
# (hipify inserts a `#include "hip/hip_runtime.h"` and does in-place token
# substitutions). Each diagnostic's hipified line is mapped back to the
# original source line with difflib before blaming, so the ownership lookup is
# exact rather than approximate.
#
# Usage
# -----
#   clang-tidy ... | clang-tidy-nccl-filter.py \
#       --repo-root <git worktree root> \
#       --rccl-src  <abs path to projects/rccl/src> \
#       --hipify-src <abs path to build/<type>/hipify/src>
#
# Reads clang-tidy output on stdin, writes the filtered output on stdout, and
# prints a one-line summary (kept / dropped) to stderr. Exit status is 1 if any
# RCCL-owned finding survived (so it can drive a gate), 0 otherwise.

import argparse
import difflib
import os
import re
import subprocess
import sys
from functools import lru_cache

# A clang-tidy diagnostic header: "path:line:col: warning|error: msg [check]".
DIAG_RE = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+): "
                     r"(?P<level>warning|error|note): ")


def git(repo_root, *args):
    return subprocess.run(
        ["git", "-C", repo_root, *args],
        check=True, capture_output=True, text=True,
    ).stdout


# Subject patterns that identify the *subtree* merge whose second parent is the
# pure NCCL-upstream content, in priority order. The canonical form used by the
# RCCL sync tooling is "merge(rccl): sync NCCL vX.Y.Z into projects/rccl";
# "Merge remote-tracking branch 'nccl/...'" covers the older manual style.
# Deliberately excluded are PR squash-merges ("rccl: sync NCCL ... (#NNNN)")
# and develop back-merges ("Merge branch 'develop' into ...nccl-sync..."),
# whose second parent is an RCCL branch rather than the NCCL side.
SYNC_MERGE_PATTERNS = (
    re.compile(r"sync\s+NCCL.*into\s+projects/rccl", re.IGNORECASE),
    re.compile(r"Merge\s+remote-tracking\s+branch\s+'nccl/", re.IGNORECASE),
)


def _merge_entries(repo_root):
    log = git(repo_root, "log", "--merges", "--format=%H %P%x00%s", "-2000")
    for entry in log.splitlines():
        head, _, subject = entry.partition("\x00")
        fields = head.split()
        commit = fields[0]
        parents = [f for f in fields[1:] if re.fullmatch(r"[0-9a-f]{40}", f)]
        yield commit, parents, subject


def detect_nccl_side(repo_root):
    """Return the commit hash of the NCCL-owned side of the latest sync merge.

    Scans merge commits newest-first and returns the second parent of the first
    one that matches a subtree-sync subject pattern and whose second parent is
    an ancestor of HEAD (so the bounded blame is valid).
    """
    entries = list(_merge_entries(repo_root))
    for pattern in SYNC_MERGE_PATTERNS:
        for commit, parents, subject in entries:
            if len(parents) >= 2 and pattern.search(subject):
                side = parents[1]
                ok = subprocess.run(
                    ["git", "-C", repo_root, "merge-base", "--is-ancestor",
                     side, "HEAD"]).returncode == 0
                if ok:
                    return side
    raise SystemExit("could not auto-detect an NCCL subtree-sync merge commit; "
                     "pass --nccl-side explicitly")


class OwnershipResolver:
    def __init__(self, repo_root, rccl_src, hipify_src, nccl_side):
        self.repo_root = repo_root
        self.rccl_src = os.path.abspath(rccl_src)
        self.hipify_src = os.path.abspath(hipify_src)
        self.nccl_side = nccl_side

    def original_path(self, hipified_path):
        """Map a hipified source path back to the original RCCL source path."""
        ap = os.path.abspath(hipified_path)
        if not ap.startswith(self.hipify_src + os.sep):
            return None
        rel = os.path.relpath(ap, self.hipify_src)
        orig = os.path.join(self.rccl_src, rel)
        return orig if os.path.isfile(orig) else None

    @lru_cache(maxsize=None)
    def _line_map(self, hipified_path, original_path):
        """Map hipified 1-based line numbers to original 1-based line numbers.

        Uses difflib on the two files; unchanged and replaced lines map
        positionally, hipify-inserted lines (no original counterpart) map to
        the nearest preceding original line.
        """
        with open(hipified_path, errors="replace") as fh:
            hip_lines = fh.read().splitlines()
        with open(original_path, errors="replace") as fh:
            orig_lines = fh.read().splitlines()
        mapping = {}
        sm = difflib.SequenceMatcher(a=hip_lines, b=orig_lines, autojunk=False)
        last_orig = 1
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            for k in range(i1, i2):
                if tag in ("equal", "replace"):
                    off = k - i1
                    oj = j1 + off
                    if oj < j2:
                        last_orig = oj + 1
                # For "insert"/"delete"/overflowed replace, fall back to the
                # last known original line so the blame lookup stays sane.
                mapping[k + 1] = last_orig
        return mapping

    def _nccl_side_relpath(self, original_path):
        """Map projects/rccl/src/<rel> to the NCCL upstream path src/<rel>.

        The subtree sync merges the NCCL tree (root layout: src/...) into
        projects/rccl, so the NCCL-owned version of projects/rccl/src/foo.cc
        lives at src/foo.cc in the NCCL-side commit.
        """
        rel = os.path.relpath(os.path.abspath(original_path), self.rccl_src)
        return f"src/{rel}"

    @lru_cache(maxsize=None)
    def _nccl_owned_lines(self, original_path):
        """Return the set of original line numbers owned entirely by NCCL.

        A line is NCCL-owned iff its content is identical to the corresponding
        upstream NCCL source at the last sync (i.e. it falls in an "equal"
        block of a difflib comparison against the NCCL-side file). Lines RCCL
        added or changed relative to upstream are RCCL-owned. This is a pure
        content diff: fast, deterministic, and independent of commit history.

        Files with no NCCL-side counterpart are RCCL-only, so every line is
        RCCL-owned (empty set returned).
        """
        nccl_blob = f"{self.nccl_side}:{self._nccl_side_relpath(original_path)}"
        try:
            upstream = subprocess.run(
                ["git", "-C", self.repo_root, "show", nccl_blob],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
        except subprocess.CalledProcessError:
            return set()  # RCCL-only file
        with open(original_path, errors="replace") as fh:
            current = fh.read().splitlines()
        nccl_lines = set()
        sm = difflib.SequenceMatcher(a=upstream, b=current, autojunk=False)
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # b-side lines j1..j2 (0-based) are unchanged from upstream.
                for j in range(j1, j2):
                    nccl_lines.add(j + 1)
        return nccl_lines

    def is_rccl_owned(self, hipified_path, hip_line):
        original = self.original_path(hipified_path)
        if original is None:
            # No upstream counterpart => an RCCL-only file => RCCL-owned.
            return True
        orig_line = self._line_map(os.path.abspath(hipified_path), original).get(hip_line)
        if orig_line is None:
            return True
        return orig_line not in self._nccl_owned_lines(original)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--rccl-src", required=True)
    ap.add_argument("--hipify-src", required=True)
    ap.add_argument("--nccl-side", default=None,
                    help="commit of the NCCL-owned side (auto-detected if omitted)")
    args = ap.parse_args()

    nccl_side = args.nccl_side or detect_nccl_side(args.repo_root)
    print(f"[nccl-filter] NCCL-owned boundary commit: {nccl_side[:12]}",
          file=sys.stderr)

    resolver = OwnershipResolver(args.repo_root, args.rccl_src,
                                 args.hipify_src, nccl_side)

    # A diagnostic is a header line followed by any number of note/context
    # lines (until the next header). Decide ownership from the header and keep
    # or drop the whole block together.
    kept = dropped = 0
    keep_block = True
    have_rccl_finding = False
    for raw in sys.stdin:
        m = DIAG_RE.match(raw)
        if m and m.group("level") in ("warning", "error"):
            owned = resolver.is_rccl_owned(m.group("path"), int(m.group("line")))
            keep_block = owned
            if owned:
                kept += 1
                have_rccl_finding = True
            else:
                dropped += 1
        if keep_block:
            sys.stdout.write(raw)

    print(f"[nccl-filter] RCCL-owned findings kept: {kept}, "
          f"NCCL-owned findings dropped: {dropped}", file=sys.stderr)
    sys.exit(1 if have_rccl_finding else 0)


if __name__ == "__main__":
    main()
