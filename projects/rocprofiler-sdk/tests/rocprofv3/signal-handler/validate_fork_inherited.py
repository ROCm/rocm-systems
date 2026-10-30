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

"""Validate that plain-fork children produce no profiler output."""

import glob
import json
import os
import re
import sqlite3
from contextlib import closing

import pytest

NUM_PREFORK_MARKERS = 5000


def load_outputs(output_dir):
    outputs = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "*_results.db"))):
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:

            def count(pattern):
                return conn.execute(
                    "SELECT COUNT(*) FROM rocpd_event WHERE extdata LIKE ?", (pattern,)
                ).fetchone()[0]

            exit_markers = [
                row[0]
                for row in conn.execute(
                    "SELECT string FROM rocpd_string WHERE string LIKE 'exit_marker %'"
                )
            ]
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
                "exit_markers": exit_markers,
                "pids": [
                    row[0] for row in conn.execute("SELECT pid FROM rocpd_info_process")
                ],
                "uuid": conn.execute(
                    "SELECT value FROM rocpd_metadata WHERE tag = 'uuid'"
                ).fetchone()[0],
            }
    return outputs


def load_json_outputs(output_dir):
    outputs = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "*_results.json"))):
        try:
            with open(path, "r", encoding="utf-8") as inp:
                process = json.load(inp)["rocprofiler-sdk-tool"][0]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            pytest.fail(f"Invalid JSON output {path}: {error}")

        marker_messages = {
            entry["key"]: entry["value"] for entry in process["strings"]["marker_api"]
        }
        marker_records = process["buffer_records"]["marker_api"]
        hip_records = process["buffer_records"]["hip_api"]
        messages = [
            marker_messages[record["correlation_id"]["internal"]]
            for record in marker_records
        ]
        process_start = process["metadata"]["init_time"]
        process_id = process["metadata"]["pid"]
        outputs[path] = {
            "parent": sum(
                msg.startswith("exit_marker parent fork-after-work") for msg in messages
            ),
            "child": sum(
                msg.startswith("exit_marker child fork-after-work") for msg in messages
            ),
            "prefork": sum(msg.startswith("prefork_marker ") for msg in messages),
            "hip": len(hip_records),
            "before_start": sum(
                record["start_timestamp"] < process_start
                for record in hip_records + marker_records
            ),
            "foreign_thread": sum(
                record["thread_id"] != process_id
                for record in hip_records + marker_records
            ),
        }
    return outputs


def test_only_parent_output_is_created(output_dir):
    outputs = load_outputs(output_dir)
    assert len(outputs) == 1, f"Expected only the parent output: {outputs}"
    path, data = next(iter(outputs.items()))
    assert data["parent"] == 1, f"{path}: expected the parent exit marker: {data}"
    assert data["child"] == 0, f"{path}: child marker leaked into parent output: {data}"
    assert len(data["exit_markers"]) == 1, data["exit_markers"]
    marker_pid = int(re.search(r" pid:(\d+)", data["exit_markers"][0]).group(1))
    assert data["pids"] == [marker_pid], (path, data)


def test_parent_reports_its_records(output_dir):
    outputs = load_outputs(output_dir)
    parents = [v for v in outputs.values() if v["parent"] == 1]
    assert len(parents) == 1, f"Expected one parent output: {outputs}"
    assert parents[0]["prefork"] == NUM_PREFORK_MARKERS, outputs
    assert parents[0]["hip"] > 0, outputs


def test_json_contains_only_parent_records(output_dir):
    outputs = load_json_outputs(output_dir)
    assert len(outputs) == 1, f"Expected only the parent JSON output: {outputs}"
    path, parent = next(iter(outputs.items()))
    assert parent["parent"] == 1, (path, parent)
    assert parent["child"] == 0, (path, parent)
    assert parent["prefork"] == NUM_PREFORK_MARKERS, outputs
    assert parent["hip"] > 0, outputs
    assert parent["before_start"] == 0, outputs
    assert parent["foreign_thread"] == 0, outputs
