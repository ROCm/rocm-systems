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

from __future__ import absolute_import

import glob
import csv
import json


def collapse_dict_list(data, key="rocprofiler-sdk-tool"):
    """Collapse a dictionary entry list into a single mapped value"""

    def check_return(_data):
        assert isinstance(_data, dict), "expected dict, type: {}".format(
            type(_data).__name__
        )
        return _data

    if (
        key in data.keys()
        and len(data.keys()) == 1
        and isinstance(data[key], (list, tuple))
        and len(data[key]) == 1
    ):
        return check_return({key: data[key][0]})

    return check_return(data)


def find_single_file(pattern, description="file"):
    """
    Find a single file matching the glob pattern.

    Args:
        pattern: Glob pattern to match (e.g., "out_*_results.csv")
        description: Human-readable description for error messages

    Returns:
        str: Path to the matched file

    Raises:
        AssertionError: If pattern doesn't match exactly one file
    """
    matches = glob.glob(pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 {description} matching {pattern}, found {len(matches)}: {matches}"
    return matches[0]


def read_csv_with_glob(filename_pattern, description="CSV file"):
    """
    Read CSV file using glob pattern to find it.

    Args:
        filename_pattern: Glob pattern to find the CSV file
        description: Human-readable description for error messages

    Returns:
        list: List of dictionaries, one per CSV row
    """
    filepath = find_single_file(filename_pattern, description)
    data = []
    with open(filepath, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


def read_json_with_glob(filename_pattern, description="JSON file"):
    """
    Read JSON file using glob pattern to find it.

    Args:
        filename_pattern: Glob pattern to find the JSON file
        description: Human-readable description for error messages

    Returns:
        dict: Parsed JSON data
    """
    filepath = find_single_file(filename_pattern, description)
    with open(filepath, "r") as inp:
        return json.load(inp)
