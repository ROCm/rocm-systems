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
Validation for processes that share one output name (no %pid% in -o).

Each process must publish its own complete output in every format instead of
replacing or truncating another process's output, and no private output may be
left behind.
"""

import glob
import json
import os
import re
import sqlite3
from contextlib import closing

import pytest

# parent + 2 forked children
NUM_PROCESSES = 3
EXIT_MARKER = re.compile(rb"exit_marker (?:parent|child) fork ppid:\d+ pid:\d+")


def outputs(output_dir, name, ext):
    """Published outputs for one name: shared_<name><ext> and shared_<name>_<pid><ext>."""
    pattern = re.compile(rf"shared_{re.escape(name)}(?:_\d+)?{re.escape(ext)}$")
    return sorted(
        path
        for path in glob.glob(os.path.join(output_dir, f"shared_{name}*{ext}"))
        if pattern.search(os.path.basename(path))
    )


def exit_markers(path):
    with open(path, "rb") as ifs:
        return set(EXIT_MARKER.findall(ifs.read()))


def assert_one_output_per_process(files):
    assert len(files) == NUM_PROCESSES, f"Expected one output per process: {files}"
    markers = []
    for path in files:
        found = exit_markers(path)
        assert len(found) == 1, f"{path}: expected one exit marker, found {found}"
        markers.extend(found)
    assert len(set(markers)) == NUM_PROCESSES, f"Outputs share a process: {markers}"


def test_rocpd(output_dir):
    files = outputs(output_dir, "results", ".db")
    for path in files:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",), path
    assert_one_output_per_process(files)


def test_json(output_dir):
    files = outputs(output_dir, "results", ".json")
    for path in files:
        with open(path) as ifs:
            json.load(ifs)
    assert_one_output_per_process(files)


def test_perfetto(output_dir):
    assert_one_output_per_process(outputs(output_dir, "results", ".pftrace"))


@pytest.mark.parametrize("name", ["marker_api_trace", "agent_info"])
def test_csv(output_dir, name):
    files = outputs(output_dir, name, ".csv")
    assert len(files) == NUM_PROCESSES, f"Expected one {name} CSV per process: {files}"
    if name == "marker_api_trace":
        assert_one_output_per_process(files)


def test_otf2(output_dir):
    anchors = outputs(output_dir, "results", ".otf2")
    assert (
        len(anchors) == NUM_PROCESSES
    ), f"Expected one OTF2 archive per process: {anchors}"
    for anchor in anchors:
        stem = anchor[: -len(".otf2")]
        assert os.path.isdir(stem) and os.listdir(stem), f"Missing trace directory {stem}"
        assert os.path.isfile(f"{stem}.def"), f"Missing definitions {stem}.def"


def test_no_private_outputs_left(output_dir):
    leftovers = glob.glob(os.path.join(output_dir, ".*.tmp*"))
    assert leftovers == [], f"Private outputs left behind: {leftovers}"
