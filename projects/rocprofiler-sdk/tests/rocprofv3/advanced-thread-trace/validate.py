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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
