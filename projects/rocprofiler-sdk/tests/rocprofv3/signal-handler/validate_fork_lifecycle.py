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

"""Validate real fork-handler regressions without accepting missing parent output."""

import glob
import json
import os

import pytest


def test_expected_process_outputs(output_dir, process_type):
    paths = sorted(glob.glob(os.path.join(output_dir, "*_results.json")))
    reattach_retry = process_type == "detach-output-retry-reattach"
    full_child_matrix = process_type == "fork-during-finalization"
    single_output = not (reattach_retry or full_child_matrix)
    expected_count = 1 if single_output else 2
    assert len(paths) == expected_count, paths
    pids = set()
    messages = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as inp:
                process = json.load(inp)["rocprofiler-sdk-tool"][0]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            pytest.fail(f"Invalid JSON output {path}: {error}")
        pid = process["metadata"]["pid"]
        assert reattach_retry or pid not in pids, paths
        pids.add(pid)
        strings = {
            entry["key"]: entry["value"] for entry in process["strings"]["marker_api"]
        }
        for record in process["buffer_records"]["marker_api"]:
            messages.append(strings[record["correlation_id"]["internal"]])
            assert record["thread_id"] == pid, (path, record)
    assert any(
        message.startswith(f"exit_marker parent {process_type}") for message in messages
    ), messages
    if reattach_retry:
        assert "exit_marker reattached session after failed detach" in messages, messages
    elif full_child_matrix:
        assert "exit_marker exec fresh initialization" in messages, messages
    assert not any(
        message.startswith("exit_marker child") for message in messages
    ), messages
