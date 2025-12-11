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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
