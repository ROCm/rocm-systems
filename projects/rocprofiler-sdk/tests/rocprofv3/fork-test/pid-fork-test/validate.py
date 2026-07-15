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

import os
import sys
import glob
import pytest


def _pid_fields(basename):
    """Return the underscore-separated fields between 'out_' and '_results.db'.
    out_<pid>_results.db -> ['<pid>']; a (buggy) double-pid out_<pid>_<pid>_results.db
    -> ['<pid>', '<pid>']."""
    prefix, suffix = "out_", "_results.db"
    assert basename.startswith(prefix) and basename.endswith(
        suffix
    ), f"Unexpected database name: {basename}"
    return basename[len(prefix) : -len(suffix)].split("_")


def get_pid_databases(output_dir):
    return glob.glob(os.path.join(output_dir, "out_*_results.db"))


def test_pid_suffix_is_suppressed(output_dir):
    """
    With -o out_%pid%, %pid% already expands to each process's own pid, so the
    filename is unique per process WITHOUT the extra _<pid> suffix that
    get_output_filename() appends to descendants by default. This test proves the
    suffix is suppressed: every database is named out_<pid>_results.db with a
    SINGLE pid field -- never out_<pid>_<pid>_results.db (a double pid would mean
    the suffix was wrongly appended on top of the already-encoded pid).
    """
    assert (
        output_dir is not None
    ), "output_dir must be provided via --output-dir when running this test."

    db_files = get_pid_databases(output_dir)

    # Parent + at least one forked/spawned child.
    assert len(db_files) >= 2, (
        f"Expected at least 2 databases (parent + child), "
        f"but found {len(db_files)}: {db_files}"
    )

    pids = []
    for db_file in db_files:
        fields = _pid_fields(os.path.basename(db_file))
        assert len(fields) == 1, (
            f"Expected a single pid field (out_<pid>_results.db) because %pid% "
            f"already encodes the pid, but {db_file} has {len(fields)} fields "
            f"{fields} -- a redundant pid suffix was appended."
        )
        assert fields[0].isdigit(), f"Expected numeric pid in {db_file}, got {fields[0]}"
        pids.append(fields[0])

    # Each process's pid is unique, so the suppressed-suffix names must not collide.
    assert len(set(pids)) == len(pids), f"Duplicate pid-named databases found: {pids}"

    # Explicitly assert no double-pid database slipped through.
    double_pid = glob.glob(os.path.join(output_dir, "out_*_*_results.db"))
    assert not double_pid, (
        f"Found double-pid database(s) {double_pid}: the _<pid> suffix was appended "
        f"even though %pid% already encodes the pid."
    )

    for db_file in db_files:
        assert os.path.exists(db_file), f"Database file {db_file} does not exist"
        assert os.path.getsize(db_file) > 0, f"Database file {db_file} is empty"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
