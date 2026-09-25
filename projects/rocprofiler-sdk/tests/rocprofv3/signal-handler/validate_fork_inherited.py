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
Validation for records made before fork(): only the parent may report them. Each fork()
child must report only its own records.
"""

import glob
import os
import sqlite3
from contextlib import closing

NUM_PREFORK_MARKERS = 5000


def load_outputs(output_dir):
    outputs = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "*_results.db"))):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            count = lambda pattern: conn.execute(
                "SELECT COUNT(*) FROM rocpd_event WHERE extdata LIKE ?", (pattern,)
            ).fetchone()[0]
            outputs[path] = {
                "parent": count('%"exit_marker parent %'),
                "child": count('%"exit_marker child %'),
                "prefork": count('%"prefork_marker %'),
                "hip": conn.execute(
                    "SELECT COUNT(*) FROM regions WHERE name IN ('hipMalloc', 'hipFree')"
                ).fetchone()[0],
                # records that started before this process did (for a child: before fork())
                "before_start": conn.execute(
                    "SELECT COUNT(*) FROM regions "
                    "WHERE start < (SELECT start FROM rocpd_info_process)"
                ).fetchone()[0],
            }
    return outputs


def test_parent_reports_its_records(output_dir):
    outputs = load_outputs(output_dir)
    parents = [v for v in outputs.values() if v["parent"] == 1]
    assert len(parents) == 1, f"Expected one parent output: {outputs}"
    assert parents[0]["prefork"] == NUM_PREFORK_MARKERS, outputs
    assert parents[0]["hip"] > 0, outputs


def test_children_report_only_their_records(output_dir):
    outputs = load_outputs(output_dir)
    children = {k: v for k, v in outputs.items() if v["child"] == 1}
    # 2 forked children
    assert len(children) == 2, f"Expected two child outputs: {outputs}"
    for path, counts in children.items():
        assert counts["prefork"] == 0, f"{path} reports the parent's markers: {counts}"
        assert counts["hip"] == 0, f"{path} reports the parent's HIP calls: {counts}"
        assert (
            counts["before_start"] == 0
        ), f"{path} reports records from before fork(): {counts}"
