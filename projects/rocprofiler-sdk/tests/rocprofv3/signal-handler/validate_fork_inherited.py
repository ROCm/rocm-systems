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
child event stream must contain only records captured after that child started.
"""

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


def test_each_process_records_its_own_identity(output_dir):
    outputs = load_outputs(output_dir)
    assert len(outputs) == 3, f"Expected one parent and two child outputs: {outputs}"

    for path, data in outputs.items():
        assert (
            len(data["exit_markers"]) == 1
        ), f"{path}: expected one exit marker, found {data['exit_markers']}"
        marker_pid = int(re.search(r" pid:(\d+)", data["exit_markers"][0]).group(1))
        assert data["pids"] == [
            marker_pid
        ], f"{path}: recorded pids {data['pids']}, process was {marker_pid}"

    uuids = [data["uuid"] for data in outputs.values()]
    assert len(set(uuids)) == 3, f"Expected distinct process UUIDs: {outputs}"


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


def test_json_outputs_contain_only_process_local_records(output_dir):
    outputs = load_json_outputs(output_dir)
    assert len(outputs) == 3, f"Expected one parent and two child JSON outputs: {outputs}"

    parents = [v for v in outputs.values() if v["parent"] == 1]
    assert len(parents) == 1, f"Expected one parent JSON output: {outputs}"
    assert parents[0]["prefork"] == NUM_PREFORK_MARKERS, outputs
    assert parents[0]["hip"] > 0, outputs

    children = {k: v for k, v in outputs.items() if v["child"] == 1}
    assert len(children) == 2, f"Expected two child JSON outputs: {outputs}"
    for path, counts in children.items():
        assert counts["prefork"] == 0, f"{path} reports the parent's markers: {counts}"
        assert (
            counts["before_start"] == 0
        ), f"{path} reports records from before fork(): {counts}"
        assert (
            counts["foreign_thread"] == 0
        ), f"{path} attributes child records to the parent thread: {counts}"
