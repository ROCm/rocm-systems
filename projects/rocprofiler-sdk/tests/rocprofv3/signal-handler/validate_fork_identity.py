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
Validation for plain fork() children: each process's rocpd database must carry that
process's own pid and a UUID distinct from every other process.
"""

import glob
import os
import re
import sqlite3
from contextlib import closing


def load_identities(output_dir):
    identities = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*_results.db"))):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            markers = [
                row[0]
                for row in conn.execute(
                    "SELECT string FROM rocpd_string WHERE string LIKE 'exit_marker %'"
                )
            ]
            pids = [row[0] for row in conn.execute("SELECT pid FROM rocpd_info_process")]
            uuid = conn.execute(
                "SELECT value FROM rocpd_metadata WHERE tag = 'uuid'"
            ).fetchone()[0]
        identities.append((path, markers, pids, uuid))
    return identities


def test_each_process_records_its_own_pid(output_dir):
    identities = load_identities(output_dir)
    # parent + 2 forked children
    assert len(identities) == 3, f"Expected one database per process: {identities}"
    for path, markers, pids, _ in identities:
        assert len(markers) == 1, f"{path}: expected one exit marker, found {markers}"
        marker_pid = int(re.search(r" pid:(\d+)", markers[0]).group(1))
        assert pids == [
            marker_pid
        ], f"{path}: recorded pids {pids}, process was {marker_pid}"


def test_each_process_has_a_distinct_uuid(output_dir):
    uuids = [uuid for _, _, _, uuid in load_identities(output_dir)]
    assert len(uuids) == 3 and len(set(uuids)) == 3, f"Duplicate rocpd UUIDs: {uuids}"
