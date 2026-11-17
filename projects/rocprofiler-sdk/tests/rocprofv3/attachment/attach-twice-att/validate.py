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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import re
import sys
from pathlib import Path

import pytest

ATT_FILENAME_PATTERN = re.compile(r"_shader_engine_\d+_(\d+)\.att$")
UI_DIRECTORY_PATTERN = re.compile(r"ui_output_agent_\d+_dispatch_(\d+)$")


def load_json(path):
    assert path.is_file(), f"missing JSON file: {path}"
    assert path.stat().st_size > 0, f"empty JSON file: {path}"
    with path.open() as inp:
        return json.load(inp)


def find_results_json(cycle_dir):
    matches = [
        path
        for path in cycle_dir.rglob("*_results.json")
        if not any(part.startswith("ui_output_agent_") for part in path.parts)
    ]

    assert len(matches) == 1, f"expected one results JSON in {cycle_dir}, found {matches}"
    return matches[0]


def get_tool_data(results_path):
    data = load_json(results_path)
    assert isinstance(data, dict), f"invalid JSON root in {results_path}"
    assert set(data) == {
        "rocprofiler-sdk-tool"
    }, f"unexpected results JSON root keys in {results_path}: {list(data)}"

    tool_data = data["rocprofiler-sdk-tool"]
    if isinstance(tool_data, list):
        assert (
            len(tool_data) == 1
        ), f"expected one tool result in {results_path}, found {len(tool_data)}"
        tool_data = tool_data[0]

    assert isinstance(tool_data, dict), f"invalid tool data in {results_path}"
    return tool_data


def iter_wave_entries(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_wave_entries(value)
    elif isinstance(node, list) and node and isinstance(node[0], str):
        yield node


def validate_decoded_output(ui_dir):
    manifest_path = ui_dir / "filenames.json"
    manifest = load_json(manifest_path)
    wave_filenames = manifest.get("wave_filenames")
    assert (
        isinstance(wave_filenames, dict) and wave_filenames
    ), f"wave_filenames is empty in {manifest_path}"

    wave_entries = list(iter_wave_entries(wave_filenames))
    assert wave_entries, f"no decoded wave entries in {manifest_path}"

    for entry in wave_entries:
        wave_path = ui_dir / entry[0]
        wave_data = load_json(wave_path)
        instructions = wave_data.get("wave", {}).get("instructions")
        assert instructions, f"no decoded instructions in {wave_path}"


def load_cycle_summary(cycle_dir):
    results_path = find_results_json(cycle_dir)
    tool_data = get_tool_data(results_path)

    metadata = tool_data.get("metadata", {})
    pid = metadata.get("pid", 0)
    assert pid > 0, f"invalid PID in {results_path}: {pid}"

    fini_time = metadata.get("fini_time", 0)
    assert fini_time > 0, f"invalid fini_time in {results_path}: {fini_time}"

    att_filenames = tool_data.get("strings", {}).get("att_filenames")
    assert (
        isinstance(att_filenames, list) and att_filenames
    ), f"att_filenames is empty in {results_path}"

    dispatch_ids = set()
    for filename in att_filenames:
        basename = Path(filename).name
        match = ATT_FILENAME_PATTERN.search(basename)
        assert match, f"unexpected ATT filename in {results_path}: {filename}"
        dispatch_ids.add(int(match.group(1)))

    return results_path, pid, fini_time, att_filenames, dispatch_ids


def validate_cycle(cycle_dir, max_dispatches):
    results_path, _, fini_time, att_filenames, dispatch_ids = load_cycle_summary(
        cycle_dir
    )

    raw_files = {path.name: path for path in cycle_dir.rglob("*.att")}
    for filename in att_filenames:
        basename = Path(filename).name
        raw_path = raw_files.get(basename)
        assert raw_path is not None, f"ATT file listed in JSON is missing: {basename}"
        assert raw_path.stat().st_size > 0, f"ATT file is empty: {raw_path}"

    assert 0 < len(dispatch_ids) <= max_dispatches, (
        f"expected between 1 and {max_dispatches} captured dispatches in {cycle_dir}, "
        f"found {sorted(dispatch_ids)}"
    )

    ui_dirs = sorted(
        path for path in cycle_dir.glob("ui_output_agent_*") if path.is_dir()
    )
    assert len(ui_dirs) == len(dispatch_ids), (
        f"expected one decoded output directory for each raw ATT dispatch in {cycle_dir}, "
        f"found {[path.name for path in ui_dirs]}"
    )

    ui_dispatch_ids = set()
    for ui_dir in ui_dirs:
        match = UI_DIRECTORY_PATTERN.fullmatch(ui_dir.name)
        assert match, f"unexpected decoded output directory: {ui_dir}"
        ui_dispatch_ids.add(int(match.group(1)))
        validate_decoded_output(ui_dir)

    assert ui_dispatch_ids == dispatch_ids, (
        f"decoded dispatches {sorted(ui_dispatch_ids)} do not match raw ATT dispatches "
        f"{sorted(dispatch_ids)} in {cycle_dir}"
    )

    print(
        f"{cycle_dir.name}: validated dispatches {sorted(dispatch_ids)}, "
        f"fini_time={fini_time}"
    )
    return fini_time, dispatch_ids


@pytest.mark.parametrize("cycle_name", ("cycle1", "cycle2"))
def test_attachment_cycle_output(output_root, max_dispatches, cycle_name):
    validate_cycle(output_root / cycle_name, max_dispatches)


def test_attachment_session_metadata(output_root):
    _, cycle1_pid, cycle1_fini, _, _ = load_cycle_summary(output_root / "cycle1")
    _, cycle2_pid, cycle2_fini, _, _ = load_cycle_summary(output_root / "cycle2")

    assert (
        cycle2_fini > cycle1_fini
    ), f"second fini_time {cycle2_fini} did not advance past first fini_time {cycle1_fini}"
    assert (
        cycle2_pid == cycle1_pid
    ), f"attachment cycles reported different PIDs: {cycle1_pid} and {cycle2_pid}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
