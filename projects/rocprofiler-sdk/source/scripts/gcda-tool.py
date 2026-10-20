#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Maintenance of gcov counter (.gcda) files for code coverage CI.

A test killed by a timeout (SIGKILL) while libgcov is writing a .gcda file
can leave it corrupt. A newly created file ends up truncated: gcov silently
under-reports it and gcov-tool merge crashes on it. The check command finds
truncated files with GCC's own tools and deletes them:

  - gcov-tool rewrite reads every file with libgcov's reader and warns about
    files that end mid-record or are not gcov data files at all.
  - gcov-dump function counts: libgcov writes a record for every function in
    the translation unit, so a .gcda must list as many functions as its .gcno.
    This catches files that are truncated on a record boundary, which
    gcov-tool rewrite accepts.

A file that is only missing part or all of its final end-of-file word passes
both checks, which is harmless: gcov and gcov-tool merge read all of its
counters.

libgcov rewrites an existing .gcda in place without truncating it, so a kill
during a rewrite leaves a well-formed file whose leading counters are from the
new run and the rest from the previous one. No check can detect this; it
undercounts the killed run's coverage.
"""

import os
import re
import sys
import shutil
import argparse
import tempfile
import subprocess
import concurrent.futures


def gcov_sibling(gcov, name):
    """Return the gcov-tool/gcov-dump installed alongside the given gcov"""
    path = shutil.which(gcov)
    if path is None:
        raise RuntimeError(f"{gcov} not found")
    path = os.path.realpath(path)
    dirname, basename = os.path.split(path)
    sibling = os.path.join(dirname, basename.replace("gcov", name, 1))
    if not os.path.isfile(sibling):
        raise RuntimeError(f"{name} matching {path} not found (expected {sibling})")
    return sibling


def find_gcda(root):
    for dirpath, _, filenames in os.walk(root):
        for itr in filenames:
            if itr.endswith(".gcda"):
                yield os.path.join(dirpath, itr)


def rewrite_errors(gcov_tool, root):
    """Files that gcov-tool rewrite reports problems with"""
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [gcov_tool, "rewrite", "-o", os.path.join(tmpdir, "out"), root],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    # gcov-tool changes into root and names files relative to it, e.g.:
    #   gcov-tool-11: warning: ./a/b.gcda:record size mismatch 2 bytes unread
    #   ./a/b.gcda:not a gcov data file
    bad = {}
    for line in proc.stdout.splitlines():
        match = re.search(r"(\./.*?\.gcda):(.*)", line)
        if match:
            path = os.path.normpath(os.path.join(root, match.group(1)))
            bad.setdefault(path, match.group(2).strip())

    if proc.returncode != 0 and not bad:
        raise RuntimeError(
            f"{gcov_tool} rewrite {root} failed ({proc.returncode}):\n{proc.stdout}"
        )

    return bad


def count_functions(gcov_dump, path):
    proc = subprocess.run(
        [gcov_dump, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )
    return proc.stdout.count(":FUNCTION ")


def function_count_error(gcov_dump, gcda):
    gcno = f"{gcda[:-5]}.gcno"
    if not os.path.isfile(gcno):
        return None

    ngcda = count_functions(gcov_dump, gcda)
    ngcno = count_functions(gcov_dump, gcno)
    if ngcda != ngcno:
        return f"has {ngcda} of {ngcno} functions"
    return None


def check(args):
    gcov_tool = gcov_sibling(args.gcov, "gcov-tool")
    gcov_dump = gcov_sibling(args.gcov, "gcov-dump")
    root = os.path.realpath(args.directory)

    bad = rewrite_errors(gcov_tool, root)

    allfiles = list(find_gcda(root))
    files = [itr for itr in allfiles if itr not in bad]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for path, error in zip(
            files, pool.map(lambda x: function_count_error(gcov_dump, x), files)
        ):
            if error is not None:
                bad[path] = error

    github = os.environ.get("GITHUB_ACTIONS", "") == "true"
    for path, error in sorted(bad.items()):
        message = f"deleting corrupt coverage data {path}: {error}"
        print(f"::warning::{message}" if github else f"warning: {message}")
        os.remove(path)

    print(f"Checked {len(allfiles)} .gcda files, deleted {len(bad)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Delete corrupt .gcda files under a directory"
    )
    check_parser.add_argument("directory", help="Directory to search for .gcda files")
    check_parser.add_argument(
        "--gcov",
        default="gcov",
        help="gcov used for coverage; gcov-tool and gcov-dump are taken from "
        "the same installation (default: %(default)s)",
    )
    check_parser.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
