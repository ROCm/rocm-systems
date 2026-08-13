#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import csv
import os
import sys
import pytest


def _read_csv(output_dir, pass_idx, filename):
    """Read a CSV emitted for a single pass (pass_<idx>/<filename>)"""
    path = os.path.join(output_dir, f"pass_{pass_idx}", filename)
    assert os.path.isfile(path), f"expected output for pass {pass_idx} is missing: {path}"
    with open(path, "r") as inp:
        return list(csv.DictReader(inp))


def _agent_info(output_dir, pass_idx):
    return _read_csv(output_dir, pass_idx, "out_agent_info.csv")


def _counter_data(output_dir, pass_idx):
    return _read_csv(output_dir, pass_idx, "out_counter_collection.csv")


def _require_counters(expected_counters):
    assert (
        expected_counters
    ), "--expected-counters must list one counter per pass (in order)"


def _validate_agent_info(agent_info, pass_label):
    """Validate the agent_info rows for a single pass"""
    assert len(agent_info) > 0, f"No agent info found in {pass_label}"

    for row in agent_info:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0


def _validate_counter_data(counter_data, expected_counter, pass_label):
    """Validate that a pass collected exactly the expected counter, with sane values"""
    assert len(counter_data) > 0, f"No counter data found in {pass_label}"

    for row in counter_data:
        assert (
            row["Counter_Name"] == expected_counter
        ), f"Expected {expected_counter} in {pass_label}, got {row['Counter_Name']}"
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0
        assert len(row["Counter_Value"]) > 0
        assert float(row["Counter_Value"]) >= 0


def test_pass_count(output_dir, expected_counters):
    """Verify one pass_* directory per expected counter, contiguously 1-indexed"""
    _require_counters(expected_counters)

    pass_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("pass_")
        and d[len("pass_") :].isdigit()
        and os.path.isdir(os.path.join(output_dir, d))
    ]

    assert (
        "pass_0" not in pass_dirs
    ), f"pass_0 should not exist (passes are 1-indexed), got {sorted(pass_dirs)}"

    pass_indices = sorted(int(d.split("_", 1)[1]) for d in pass_dirs)
    assert pass_indices == list(
        range(1, len(expected_counters) + 1)
    ), f"Pass directories must be contiguously 1-indexed starting at pass_1, got {pass_indices}"


def test_agent_info(output_dir, expected_counters):
    """Validate the agent info emitted for every pass"""
    _require_counters(expected_counters)

    for pass_idx in range(1, len(expected_counters) + 1):
        _validate_agent_info(_agent_info(output_dir, pass_idx), f"pass {pass_idx}")


def test_counters(output_dir, expected_counters):
    """Validate that pass_i collected exactly expected_counters[i-1]"""
    _require_counters(expected_counters)

    for pass_idx, counter in enumerate(expected_counters, start=1):
        _validate_counter_data(
            _counter_data(output_dir, pass_idx), counter, f"pass {pass_idx}"
        )


def test_same_dispatches_all_passes(output_dir, expected_counters):
    """Verify every pass collected data for the same kernel dispatches"""
    _require_counters(expected_counters)

    dispatch_ids_by_pass = {
        pass_idx: set(
            int(row["Dispatch_Id"]) for row in _counter_data(output_dir, pass_idx)
        )
        for pass_idx in range(1, len(expected_counters) + 1)
    }

    reference = dispatch_ids_by_pass.pop(1)
    for pass_idx, dispatch_ids in dispatch_ids_by_pass.items():
        assert dispatch_ids == reference, (
            f"pass {pass_idx} has different dispatch IDs than pass 1. "
            f"pass 1: {sorted(reference)}, pass {pass_idx}: {sorted(dispatch_ids)}"
        )


def test_counter_separation(output_dir, expected_counters):
    """Verify no counter appears in more than one pass"""
    _require_counters(expected_counters)

    seen = set()
    for pass_idx in range(1, len(expected_counters) + 1):
        names = set(row["Counter_Name"] for row in _counter_data(output_dir, pass_idx))
        overlap = seen & names
        assert (
            not overlap
        ), f"Counters should not overlap between passes; {overlap} seen again in pass {pass_idx}"
        seen |= names


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
