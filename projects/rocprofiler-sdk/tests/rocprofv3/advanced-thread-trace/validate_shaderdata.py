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
