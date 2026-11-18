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
Validation for signal handler integration tests.

Every scenario must produce valid and complete JSON output. The profiler intercepts
signals to flush cleanly, so a truncated or invalid JSON is a bug in signal handler.

The per-scenario expectation is passed explicitly via --expect:
  clean-exit      : exit_marker present in JSON (proves clean atexit finalization). Used
                    when the app handles signals and rocprofv3 doesn't (--disable-signal-handlers)
  flushed-markers : valid JSON with >10 marker entries (proves the profiler flushed before
                    the process died on the signal). Covers both the "app doesn't handle
                    signals" case and the coordinated-shutdown case. For the latter,
                    simply reaching this validator proves there was no deadlock (a
                    regression trips the execute-step ctest TIMEOUT instead).
"""

import json
import os
import glob

import pytest


def find_json_files(output_dir):
    """Find all JSON result files in the output directory."""
    pattern = os.path.join(output_dir, "*.json")
    return sorted(glob.glob(pattern))


def load_json(path):
    """Load and validate a JSON file. Raises on truncated/corrupt JSON."""
    with open(path, "r") as f:
        return json.load(f)


def count_markers_in_json(data):
    """Count marker events in the rocprofv3 JSON structure."""
    count = 0
    try:
        for tool_entry in data.get("rocprofiler-sdk-tool", []):
            buffer_records = tool_entry.get("buffer_records", {})
            count += len(buffer_records.get("marker_api", []))
            callback_records = tool_entry.get("callback_records", {})
            count += len(callback_records.get("marker_api", []))
    except (AttributeError, TypeError):
        pass
    return count


def test_output_files_exist(output_dir):
    """JSON output files must exist."""
    files = find_json_files(output_dir)
    assert len(files) > 0, f"No JSON output files found in {output_dir}"


def test_json_is_valid(output_dir):
    """All JSON output files must be valid (not truncated)."""
    files = find_json_files(output_dir)
    assert len(files) > 0, f"No JSON files in {output_dir}"

    for path in files:
        try:
            load_json(path)
        except json.JSONDecodeError as e:
            assert False, (
                f"JSON file is truncated/corrupt: {path}\n"
                f"Error: {e}\n"
                f"This means the profiler did not flush cleanly before process death."
            )


def test_clean_exit(output_dir, expect, process_type):
    """clean-exit: the app-specific exit_marker must be present (proves the app reached
    atexit and finalized cleanly)."""
    if expect != "clean-exit":
        pytest.skip(f"expectation is '{expect}', not clean-exit")

    files = find_json_files(output_dir)
    assert len(files) > 0

    expected_marker = f"exit_marker parent {process_type}"

    all_content = ""
    for path in files:
        with open(path, "r") as f:
            all_content += f.read()

    assert expected_marker in all_content, (
        f"Expected marker '{expected_marker}' not found in output files in {output_dir}. "
        f"App did not exit cleanly (atexit finalization failed)."
    )


def test_flushed_markers(output_dir, expect):
    """flushed-markers: JSON must contain >10 marker entries, proving the profiler flushed
    before the process died on the signal. Covers the "app doesn't handle signals" case and
    the coordinated-shutdown case (reaching this validator at all already proves the
    coordinated case did not deadlock -- a regression trips the execute-step ctest
    TIMEOUT). exit_marker is intentionally NOT required here."""
    if expect != "flushed-markers":
        pytest.skip(f"expectation is '{expect}', not flushed-markers")

    files = find_json_files(output_dir)
    assert len(files) > 0

    total_markers = 0
    for path in files:
        data = load_json(path)
        total_markers += count_markers_in_json(data)

    assert total_markers > 10, (
        f"Expected >10 marker events in JSON output, got {total_markers}. "
        f"Profiler may not have flushed marker data before signal death. "
        f"Files: {files}"
    )
