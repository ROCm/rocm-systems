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

The move and merge commands let CI collect the counters of separate test runs
and combine them into a single report without running the tests again.

A test killed by a timeout (SIGKILL) while libgcov is writing a .gcda file
can leave it truncated. gcov reads what is left, but gcov-tool merge can crash
on it. Rather than predicting which files gcov-tool accepts, merge combines
each file separately. When gcov-tool fails, it cannot tell which input is bad,
so merge keeps the largest input rather than none.

A .gcda whose stamp differs from its .gcno was written by an older compile of
the object. gcov reports such files as 0% covered without failing, so check
fails on them: the counters were collected against a build that no longer
exists, which is a problem with the CI steps, not with a test.
"""

import os
import sys
import struct
import shutil
import argparse
import tempfile
import contextlib
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
    if sibling == path:
        raise RuntimeError(f"{path} is not named like gcov: cannot find its {name}")
    if not os.path.isfile(sibling):
        raise RuntimeError(f"{name} matching {path} not found (expected {sibling})")
    return sibling


def find_gcda(root):
    for dirpath, _, filenames in os.walk(root):
        for itr in filenames:
            if itr.endswith(".gcda"):
                yield os.path.join(dirpath, itr)


def github_message(kind, message):
    github = os.environ.get("GITHUB_ACTIONS", "") == "true"
    print(f"::{kind}::{message}" if github else f"{kind}: {message}")


def read_stamp(path, magic):
    """Stamp of a .gcda or .gcno file, or None if the file has no gcov header"""
    # the header is the magic, version and stamp words, in the writer's byte order
    with open(path, "rb") as f:
        header = f.read(12)
    if len(header) < 12:
        return None
    for order in "<>":
        words = struct.unpack(f"{order}3I", header)
        if words[0] == struct.unpack(">I", magic)[0]:
            return words[2]
    return None


def stale_error(gcda):
    """Error message if a .gcda does not match the stamp of its .gcno"""
    gcno = f"{gcda[:-5]}.gcno"
    if not os.path.isfile(gcno):
        return None

    gcda_stamp = read_stamp(gcda, b"gcda")
    gcno_stamp = read_stamp(gcno, b"gcno")
    # a .gcda without a readable stamp was truncated, not built against another .gcno
    if gcda_stamp is None or gcda_stamp == gcno_stamp:
        return None
    return f"stamp {gcda_stamp} does not match {gcno} stamp {gcno_stamp}"


def check(args):
    root = os.path.realpath(args.directory)

    files = list(find_gcda(root))
    errors = [stale_error(itr) for itr in files]

    stale = sorted((path, err) for path, err in zip(files, errors) if err is not None)
    for path, error in stale:
        github_message(
            "error", f"coverage data {path} is from a different build: {error}"
        )

    print(f"Checked {len(files)} .gcda files")

    if stale:
        print(f"{len(stale)} .gcda files do not match the build", file=sys.stderr)
        return 1


def move_gcda(source, destination):
    files = list(find_gcda(source))
    for itr in files:
        target = os.path.join(destination, os.path.relpath(itr, source))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(itr, target)
    return len(files)


def move(args):
    count = move_gcda(args.source, args.destination)
    print(f"Moved {count} .gcda files from {args.source} to {args.destination}")
    if count == 0:
        # no tests ran, e.g. the build failed
        print(f"No .gcda files in {args.source}", file=sys.stderr)
        return 1


def merge_one(gcov_tool, tmpdir, relpath, inputs, output):
    """Merge one .gcda from several directories into output.

    Returns a warning if an input was left out, or the .gcda could not be written.
    """
    target = os.path.join(output, relpath)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if len(inputs) == 1:
        shutil.move(os.path.join(inputs[0], relpath), target)
        return None

    # gcov-tool merge combines directories: give it directories holding only this file
    workdir = tempfile.mkdtemp(dir=tmpdir)
    name = os.path.basename(relpath)
    result, result_inputs = None, []
    warnings = []
    for idx, itr in enumerate(inputs):
        current = os.path.join(workdir, f"in{idx}")
        os.makedirs(current)
        shutil.move(os.path.join(itr, relpath), os.path.join(current, name))
        if result is None:
            result, result_inputs = current, [itr]
            continue

        merged = os.path.join(workdir, f"out{idx}")
        proc = subprocess.run(
            [gcov_tool, "merge", "-o", merged, result, current],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if proc.returncode == 0:
            result, result_inputs = merged, result_inputs + [itr]
            continue

        # gcov-tool cannot say which input is bad: keep the larger one, not neither
        first = proc.stdout.strip().splitlines()[:1]
        result_size = os.path.getsize(os.path.join(result, name))
        if os.path.getsize(os.path.join(current, name)) > result_size:
            result, result_inputs, left_out = current, [itr], result_inputs
        else:
            left_out = [itr]
        warnings.append(
            f"gcov-tool merge failed ({proc.returncode}): {' '.join(first)}; "
            f"left out the counters from {', '.join(left_out)}"
        )

    shutil.move(os.path.join(result, name), target)
    return "; ".join(warnings) or None


def merge(args):
    gcov_tool = gcov_sibling(args.gcov, "gcov-tool")

    inputs = {}
    for itr in args.directories:
        for path in find_gcda(itr):
            inputs.setdefault(os.path.relpath(path, itr), []).append(itr)

    if not inputs:
        # no tests ran, e.g. the build failed: a report would show 0% coverage
        print("No .gcda files to merge", file=sys.stderr)
        return 1

    def merge_or_warn(tmpdir, relpath, dirs):
        try:
            return merge_one(gcov_tool, tmpdir, relpath, dirs, args.output)
        except OSError as err:
            # a failed move can leave a partial copy behind
            with contextlib.suppress(OSError):
                os.remove(os.path.join(args.output, relpath))
            return f"leaving out the file: {err}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            warnings = list(pool.map(lambda x: merge_or_warn(tmpdir, *x), inputs.items()))

    for relpath, warning in sorted(zip(inputs.keys(), warnings)):
        if warning is not None:
            github_message("warning", f"coverage data {relpath}: {warning}")

    merged = sum(os.path.isfile(os.path.join(args.output, itr)) for itr in inputs)
    print(
        f"Merged {len(args.directories)} directories into "
        f"{merged} .gcda files in {args.output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Fail if .gcda files under a directory do not match their .gcno"
    )
    check_parser.add_argument("directory", help="Directory to search for .gcda files")
    check_parser.set_defaults(func=check)

    move_parser = subparsers.add_parser(
        "move", help="Move .gcda files to another directory, keeping relative paths"
    )
    move_parser.add_argument("source", help="Directory to move .gcda files from")
    move_parser.add_argument("destination", help="Directory to move .gcda files to")
    move_parser.set_defaults(func=move)

    merge_parser = subparsers.add_parser(
        "merge",
        help="Sum the counters of .gcda files in several directories and move the "
        "result into an output directory, keeping the largest input of files "
        "gcov-tool cannot merge",
    )
    merge_parser.add_argument("-o", "--output", required=True, help="Output directory")
    merge_parser.add_argument(
        "directories", nargs="+", help="Directories of .gcda files to merge"
    )
    merge_parser.set_defaults(func=merge)

    merge_parser.add_argument(
        "--gcov",
        default="gcov",
        help="gcov used for coverage; gcov-tool is taken from the same installation "
        "(default: %(default)s)",
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
