#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import csv
import sys
import pytest
import re
import os
import glob
import json
from pathlib import Path


def test_json_data(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    strings = data["strings"]
    assert "att_filenames" in strings.keys()
    att_files = data["strings"]["att_filenames"]
    assert len(att_files) > 0


def test_code_object_memory(code_object_file_path, json_data, output_path):

    data = json_data["rocprofiler-sdk-tool"]
    tool_memory_load = data["strings"]["code_object_snapshot_filenames"]
    gfx_pattern = "gfx[a-z0-9]+"
    match = re.search(gfx_pattern, tool_memory_load[1])
    assert match != None
    gpu_name = match.group(0)

    read_bytes = lambda filename: open(os.path.join(output_path, filename), "rb").read()
    # Loads all saved code objects
    tool_memory = [read_bytes(saved) for saved in tool_memory_load[1:]]

    found = False
    for hsa_file in code_object_file_path["hsa_memory_load"]:

        m = re.search(gfx_pattern, hsa_file)
        assert m != None
        gpu = m.group(0)

        if gpu == gpu_name:
            found = True
            hsa_memory_bytes = open(hsa_file, "rb").read()
            # Checks if hsa_file is one of the saved code objects
            assert any([hsa_memory_bytes == fs for fs in tool_memory])
            break
    assert found == True


def test_perfcounter_target_cu(output_path, request):
    """
    Test that when --att-perfcounter-target-cu is specified, all perfcounter
    events are tagged only for the target CU (or no events at all).
    """
    # Get the target CU from pytest command line if provided
    target_cu = request.config.getoption("--target-cu", default=None)

    if target_cu is None:
        pytest.skip("--target-cu not specified, skipping perfcounter target CU test")

    target_cu = int(target_cu)

    # Find all perfcounter JSON files
    pattern = os.path.join(output_path, "ui_output_*", "se*_perfcounter.json")
    perfcounter_files = glob.glob(pattern)
    print(perfcounter_files)

    # It's acceptable to have no perfcounter files (target CU may not have any events)
    for pc_file in perfcounter_files:
        with open(pc_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        data = json_data.get("data", [])

        # Check that all events are for the target CU
        for event in data:
            # CU is at index 5
            event_cu = event[5]
            assert event_cu == target_cu, (
                f"Found perfcounter event for CU {event_cu}, "
                f"but target CU is {target_cu} in file {pc_file}"
            )


def test_realtime_clock(output_path):

    def verify_sorted(timestamps):

        # Sort by shader_clock (index 0)
        timestamps_sorted = sorted(timestamps, key=lambda ts: ts[0])
        # Ensure realtime clock is non descreasing
        assert all(
            curr[1] >= prev[1]
            for prev, curr in zip(timestamps_sorted, timestamps_sorted[1:])
        )

    def verify_gfxclock(timestamps, rt_frequency):

        delta_shader_clock = timestamps[-1][0] - timestamps[0][0]
        delta_realtime_ts = timestamps[-1][1] - timestamps[0][1]
        gfxclock = rt_frequency * delta_shader_clock / delta_realtime_ts

        # gfxclock must be positive
        assert gfxclock > 0
        # gfxclock must be <10GHz
        assert gfxclock < 1e10

    pattern = os.path.join(output_path, "ui_output_*", "realtime.json")
    for rt_file in glob.glob(pattern):
        with open(rt_file, "r", encoding="utf-8") as f:
            json_file = json.load(f)

        frequency = json_file["metadata"]["frequency"]
        # frequency = 0 means aqlprofile is not instrumented
        if frequency > 0:
            for key, value in json_file.items():
                # Exclude metadata and single-clock timestamps
                if "metadata" not in key and len(value) >= 2:
                    verify_sorted(value)
                    verify_gfxclock(value, frequency)


def test_other_simd_data(att_other_simd_out_dir_path):

    def find_ui_output_dirs(att_other_simd_out_dir_path):
        matches = [
            p
            for p in Path(att_other_simd_out_dir_path).glob("ui_output_agent_*")
            if p.is_dir()
        ]
        return matches

    def find_other_simd_files(other_simd_files_path):
        root = Path(other_simd_files_path)
        if not root.is_dir():
            return []
        matches = [p for p in root.glob("other_simd_se*") if p.is_file()]
        return matches

    att_ui_dispatch_dirs = find_ui_output_dirs(att_other_simd_out_dir_path)
    # more than 1 dirs must be generated by rocprofv3.
    assert len(att_ui_dispatch_dirs) > 0, "ui_output_agent_* dirs not found."

    for ui_dispatch_dir in att_ui_dispatch_dirs:
        with open(ui_dispatch_dir / "filenames.json", "r") as inp:
            filenames_json = json.load(inp)

        listed_file_names = filenames_json["other_simd_filenames"]
        assert len(listed_file_names) > 0, "other_simd_filenames is empty."

        other_simd_files_found = find_other_simd_files(ui_dispatch_dir)

        total_listed_files = 0
        for se, files in listed_file_names.items():
            assert len(files) > 0, f"other_simd_filenames[{se}] is empty."
            total_listed_files += len(files)

        assert (
            len(other_simd_files_found) == total_listed_files
        ), "other_simd files mismatch between filenames.json and files present in dir."

        for files in listed_file_names.values():
            for file in files:
                assert (
                    len(file) == 3
                ), "other_simd_filenames entry must be [filename, begin, end]."
                with open(ui_dispatch_dir / file[0], "r") as inp:
                    other_simd_file_data = json.load(inp)

                assert (
                    file[1] == other_simd_file_data["begin_time"]
                ), "begin time mismatch filenames.json and other_simd_se*_*.json"

                assert (
                    file[2] == other_simd_file_data["end_time"]
                ), "end time mismatch filenames.json and other_simd_se*_*.json"

                assert (
                    other_simd_file_data["instructions_count"] > 0
                ), "other simd instructions is empty."

                other_simd_instructions = other_simd_file_data["instructions"]

                assert (
                    len(other_simd_instructions)
                    == other_simd_file_data["instructions_count"]
                )

                last_known_time = other_simd_instructions[0][0]
                last_known_duration = other_simd_instructions[0][1]

                assert last_known_time == other_simd_file_data["begin_time"]

                # Start from the second element
                for other_simd_instruction in other_simd_instructions[1:]:
                    assert (
                        other_simd_instruction[0] >= last_known_time
                    ), "data from other_simd file is not in increasing time, corrupted data."
                    last_known_time = other_simd_instruction[0]
                    last_known_duration = other_simd_instruction[1]

                assert (
                    last_known_time + last_known_duration
                    == other_simd_file_data["end_time"]
                )


def test_shaderdata(att_shaderdata_out_dir_path):
    expected_value = 3735928559  # m0 value from kernel_lds.cpp (0xDEADBEEF)

    def find_ui_output_dirs(att_out_dir_path):
        matches = [
            p for p in Path(att_out_dir_path).glob("ui_output_agent_*") if p.is_dir()
        ]
        return matches

    def find_shaderdata_files(shaderdata_files_path):
        root = Path(shaderdata_files_path)
        if not root.is_dir():
            return []
        matches = [p for p in root.glob("shaderdata_*") if p.is_file()]
        return matches

    att_ui_dispatch_dirs = find_ui_output_dirs(att_shaderdata_out_dir_path)
    assert len(att_ui_dispatch_dirs) > 0, "ui_output_agent_* dirs not found."

    found_shaderdata = False
    for ui_dispatch_dir in att_ui_dispatch_dirs:
        with open(ui_dispatch_dir / "filenames.json", "r") as inp:
            filenames_json = json.load(inp)

        listed_file_names = filenames_json.get("shaderdata_filenames", {})
        if not listed_file_names:
            continue

        found_shaderdata = True
        shaderdata_files_found = find_shaderdata_files(ui_dispatch_dir)
        listed_count = sum(len(files) for files in listed_file_names.values())

        assert (
            len(shaderdata_files_found) == listed_count
        ), "shaderdata files mismatch between filenames.json and files present in dir."

        for files in listed_file_names.values():
            for file in files:
                with open(ui_dispatch_dir / file[0], "r") as inp:
                    shaderdata_file_data = json.load(inp)

                assert (
                    file[1] == shaderdata_file_data["begin_time"]
                ), "begin time mismatch filenames.json and shaderdata_*.json"

                assert (
                    file[2] == shaderdata_file_data["end_time"]
                ), "end time mismatch filenames.json and shaderdata_*.json"

                assert (
                    shaderdata_file_data["records_count"] > 0
                ), "shaderdata records are empty."

                shaderdata_records = shaderdata_file_data["records"]

                assert len(shaderdata_records) == shaderdata_file_data["records_count"]

                # Validate ordering and sentinel value in records.
                last_known_time = shaderdata_records[0][0]
                assert last_known_time == shaderdata_file_data["begin_time"]

                for record in shaderdata_records:
                    assert (
                        record[1] == expected_value
                    ), "shaderdata record value mismatch."

                for record in shaderdata_records[1:]:
                    assert (
                        record[0] >= last_known_time
                    ), "data from shaderdata file is not in increasing time."
                    last_known_time = record[0]

                assert (
                    last_known_time == shaderdata_file_data["end_time"]
                ), "end time mismatch between records and shaderdata_*.json"

    # Require at least one ui_output_agent_* directory with shaderdata data.
    assert found_shaderdata, "No ui_output_agent_* directory contains shaderdata data."


def test_shaderdata(att_shaderdata_out_dir_path):
    expected_value = 3735928559  # m0 value from kernel_lds.cpp (0xDEADBEEF)

    def find_ui_output_dirs(att_out_dir_path):
        matches = [
            p for p in Path(att_out_dir_path).glob("ui_output_agent_*") if p.is_dir()
        ]
        return matches

    def find_shaderdata_files(shaderdata_files_path):
        root = Path(shaderdata_files_path)
        if not root.is_dir():
            return []
        matches = [p for p in root.glob("shaderdata_*") if p.is_file()]
        return matches

    att_ui_dispatch_dirs = find_ui_output_dirs(att_shaderdata_out_dir_path)
    assert len(att_ui_dispatch_dirs) > 0, "ui_output_agent_* dirs not found."

    found_shaderdata = False
    for ui_dispatch_dir in att_ui_dispatch_dirs:
        with open(ui_dispatch_dir / "filenames.json", "r") as inp:
            filenames_json = json.load(inp)

        listed_file_names = filenames_json.get("shaderdata_filenames", {})
        if not listed_file_names:
            continue

        found_shaderdata = True
        shaderdata_files_found = find_shaderdata_files(ui_dispatch_dir)
        listed_count = sum(len(files) for files in listed_file_names.values())

        assert (
            len(shaderdata_files_found) == listed_count
        ), "shaderdata files mismatch between filenames.json and files present in dir."

        for files in listed_file_names.values():
            for file in files:
                with open(ui_dispatch_dir / file[0], "r") as inp:
                    shaderdata_file_data = json.load(inp)

                assert (
                    file[1] == shaderdata_file_data["begin_time"]
                ), "begin time mismatch filenames.json and shaderdata_*.json"

                assert (
                    file[2] == shaderdata_file_data["end_time"]
                ), "end time mismatch filenames.json and shaderdata_*.json"

                assert (
                    shaderdata_file_data["records_count"] > 0
                ), "shaderdata records are empty."

                shaderdata_records = shaderdata_file_data["records"]

                assert len(shaderdata_records) == shaderdata_file_data["records_count"]

                # Validate ordering and sentinel value in records.
                last_known_time = shaderdata_records[0][0]
                assert last_known_time == shaderdata_file_data["begin_time"]

                for record in shaderdata_records:
                    assert (
                        record[1] == expected_value
                    ), "shaderdata record value mismatch."

                for record in shaderdata_records[1:]:
                    assert (
                        record[0] >= last_known_time
                    ), "data from shaderdata file is not in increasing time."
                    last_known_time = record[0]

                assert (
                    last_known_time == shaderdata_file_data["end_time"]
                ), "end time mismatch between records and shaderdata_*.json"

    # Require at least one ui_output_agent_* directory with shaderdata data.
    assert found_shaderdata, "No ui_output_agent_* directory contains shaderdata data."


def test_multi_gpu_separate_agents(att_multi_gpu_out_dir_path):
    """
    When multiple GPUs are traced for the same dispatch, each agent must get
    its own ui_output_agent_* directory.  Before the fix, all agents' ATT data
    was merged into a single directory, causing shaderdata timestamp resets
    (each GPU has its own independent SQTT counter) and silently dropping
    occupancy data from all but the first agent.
    """

    out_dir = Path(att_multi_gpu_out_dir_path)

    # Use agent_info.csv to determine how many GPUs are on this machine.
    # The file is inside a hostname subdirectory: <out_dir>/<hostname>/<pid>_agent_info.csv
    agent_csvs = list(out_dir.glob("**/*_agent_info.csv"))
    assert agent_csvs, (
        f"No *_agent_info.csv found under {out_dir}. "
        f"Ensure the execute test runs with --output-format csv."
    )
    with open(agent_csvs[0], "r") as f:
        gpu_count = sum(1 for row in csv.DictReader(f) if row["Agent_Type"] == "GPU")
    if gpu_count < 2:
        pytest.skip(f"Only {gpu_count} GPU(s) on this machine, need >= 2")

    ui_dirs = sorted(p for p in out_dir.glob("ui_output_agent_*") if p.is_dir())

    # With --att-gpu-index 0,1 we expect at least two output directories
    # (one per agent).
    assert len(ui_dirs) >= 2, (
        f"Expected at least 2 ui_output_agent_* directories (one per GPU), "
        f"found {len(ui_dirs)}.  ATT data from multiple agents may be merged."
    )

    # Extract agent ids from directory names and verify they are distinct.
    agent_ids = set()
    pattern = re.compile(r"ui_output_agent_(\d+)_dispatch_(\d+)")
    for d in ui_dirs:
        m = pattern.search(d.name)
        assert m, f"Unexpected directory name format: {d.name}"
        agent_ids.add(m.group(1))

    assert len(agent_ids) >= 2, (
        f"Expected directories for at least 2 distinct agents, "
        f"found agent ids: {agent_ids}"
    )

    # For each directory that has shaderdata, verify timestamps are
    # monotonically increasing across all chunks (no resets from other GPUs).
    for ui_dir in ui_dirs:
        filenames_path = ui_dir / "filenames.json"
        if not filenames_path.exists():
            continue

        with open(filenames_path, "r") as f:
            filenames_json = json.load(f)

        shaderdata_filenames = filenames_json.get("shaderdata_filenames", {})
        if not shaderdata_filenames:
            continue

        for se, files in shaderdata_filenames.items():
            prev_end_time = -1
            for file_entry in files:
                begin_time = file_entry[1]
                end_time = file_entry[2]

                # Each chunk's begin_time must be >= the previous chunk's
                # end_time.  A reset (begin < prev_end) indicates data from a
                # different GPU was mixed in.
                assert begin_time >= prev_end_time, (
                    f"Shaderdata timestamp reset in {ui_dir.name} SE {se}: "
                    f"chunk {file_entry[0]} begins at {begin_time} but "
                    f"previous chunk ended at {prev_end_time}.  "
                    f"Data from multiple agents may be merged."
                )
                prev_end_time = end_time


def test_att_marker_trace(json_data, att_marker_trace_out_dir_path):
    """Verify marker-controlled ATT traced only kernels between Resume and Pause.

    Test binary launches 4 kernels:
    - before_trace_kernel: before roctxProfilerResume (should NOT be traced)
    - traced_kernel_first: after roctxProfilerResume (should be traced)
    - traced_kernel_second: after roctxProfilerResume (should be traced)
    - after_trace_kernel: after roctxProfilerPause (should NOT be traced)

    Validation parses the stats_*.csv files produced by the ATT decoder.
    Kernel names appear as instruction rows with "; <mangled_name>" and the
    demangled name in the Source column.
    """
    data = json_data["rocprofiler-sdk-tool"]
    strings = data["strings"]

    # Verify ATT data was produced
    assert "att_filenames" in strings.keys(), "No att_filenames in output"
    att_files = strings["att_filenames"]
    assert len(att_files) > 0, "Expected ATT data from marker-controlled thread trace"

    # Verify decoded ATT output directories exist
    att_ui_dirs = [
        p
        for p in Path(att_marker_trace_out_dir_path).glob("ui_output_agent_*")
        if p.is_dir()
    ]
    assert len(att_ui_dirs) > 0, "No ui_output_agent_* directories found"

    # Parse stats_*.csv files for kernel names.
    # stats_*.csv are written to the PARENT of ui_output_* dirs (see code.cpp).
    # The CSV has rows where the Instruction column is "; <mangled_name>" for
    # kernel entry points and the Source column holds the demangled name.
    traced_kernel_names = set()
    stats_files = list(Path(att_marker_trace_out_dir_path).glob("stats_*.csv"))
    assert (
        len(stats_files) > 0
    ), f"No stats_*.csv files found in {att_marker_trace_out_dir_path}"
    for stats_file in stats_files:
        with open(stats_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            assert header is not None, f"Empty stats CSV: {stats_file}"
            for row in reader:
                # Instruction column (index 2) starts with "; " for kernel names
                if len(row) >= 3 and row[2].startswith("; "):
                    traced_kernel_names.add(row[2][2:].strip())
                # Also check demangled name in Source column (index 7)
                if len(row) >= 8 and row[7].strip():
                    traced_kernel_names.add(row[7].strip())

    assert len(traced_kernel_names) > 0, "No kernel names found in stats CSV files"

    # Verify traced_kernel_first and traced_kernel_second were traced
    for expected in ("traced_kernel_first", "traced_kernel_second"):
        found = any(expected in name for name in traced_kernel_names)
        assert found, (
            f"Expected '{expected}' to be in ATT trace but it was not. "
            f"Traced kernels: {traced_kernel_names}"
        )

    # Verify before_trace_kernel and after_trace_kernel were NOT traced
    for not_expected in ("before_trace_kernel", "after_trace_kernel"):
        found = any(not_expected in name for name in traced_kernel_names)
        assert not found, (
            f"Expected '{not_expected}' to NOT be in ATT trace but it was. "
            f"Traced kernels: {traced_kernel_names}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
