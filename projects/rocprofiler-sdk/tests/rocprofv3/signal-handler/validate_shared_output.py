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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Validation for processes that share one rocpd output name (no %pid% in -o).

Each process must publish its own complete database instead of unlinking another
process's open database, and no private temporary databases may be left behind.
"""

import glob
import os
import sqlite3
from contextlib import closing


def test_each_process_keeps_a_complete_database(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "shared_results*.db")))
    # parent + 2 forked children
    assert len(files) == 3, f"Expected one database per process, found: {files}"

    # Each process emits an exit marker carrying its own pid; every database must hold
    # exactly one, from a different process.
    markers = []
    for path in files:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",), path
            found = [
                row[0]
                for row in conn.execute(
                    "SELECT string FROM rocpd_string WHERE string LIKE 'exit_marker %'"
                )
            ]
            assert len(found) == 1, f"{path}: expected one exit marker, found {found}"
            markers.extend(found)
    assert len(set(markers)) == 3, f"Expected three distinct processes, found: {markers}"


def test_no_temporary_databases_left(output_dir):
    leftovers = glob.glob(os.path.join(output_dir, ".*.tmp*"))
    assert leftovers == [], f"Temporary databases left behind: {leftovers}"
