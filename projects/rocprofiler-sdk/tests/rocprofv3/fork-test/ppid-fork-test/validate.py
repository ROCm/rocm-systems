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


def _numeric_fields(basename, prefix, suffix):
    """Return the underscore-separated fields between prefix and suffix, or None
    if the name doesn't match. For out_<A>_results.db this is ['<A>']; for
    out_<A>_<B>_results.db it is ['<A>', '<B>']."""
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        return None
    return basename[len(prefix) : -len(suffix)].split("_")


def get_pid_databases(output_dir):
    """Find rocpd database files and split them by how many numeric fields the
    name carries. The root keeps out_<ppid> (one field: the expanded %ppid%); a
    descendant gets its own pid appended -> out_<ppid>_<pid> (two fields)."""
    files = glob.glob(os.path.join(output_dir, "out_*_results.db"))
    root_files = []
    child_files = []
    for f in files:
        fields = _numeric_fields(os.path.basename(f), "out_", "_results.db")
        assert fields is not None, f"Unexpected database name: {f}"
        assert all(
            x.isdigit() for x in fields
        ), f"Expected only numeric fields (expanded %ppid%/pid) in {f}, got {fields}"
        if len(fields) == 1:
            root_files.append(f)
        elif len(fields) == 2:
            child_files.append(f)
        else:
            raise AssertionError(
                f"Database {f} has {len(fields)} numeric fields; expected 1 (root) or 2 (child)"
            )
    return root_files, child_files


def test_ppid_does_not_suppress_pid_suffix(output_dir):
    """
    With -o out_%ppid%, %ppid% expands to the PARENT process's pid -- never the
    process's own pid -- so it must NOT be mistaken for a token that already
    encodes this process's pid. The root writes out_<ppid>_results.db and each
    forked/spawned descendant must still get its OWN pid appended
    (out_<ppid>_<pid>_results.db), so no process overwrites another.
    """
    assert (
        output_dir is not None
    ), "output_dir must be provided via --output-dir when running this test."

    root_files, child_files = get_pid_databases(output_dir)

    # Root keeps out_<ppid>; descendants get their own pid suffix appended.
    assert len(root_files) == 1, (
        f"Expected exactly 1 root database (out_<ppid>_results.db), "
        f"but found {len(root_files)}: {root_files}"
    )
    assert len(child_files) >= 1, (
        f"Expected at least 1 descendant database with its own pid suffix "
        f"(out_<ppid>_<pid>_results.db), but found {len(child_files)}: {child_files}"
    )
    db_files = root_files + child_files

    # Verify all database files exist and are non-empty
    for db_file in db_files:
        assert os.path.exists(db_file), f"Database file {db_file} does not exist"
        assert os.path.getsize(db_file) > 0, f"Database file {db_file} is empty"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
