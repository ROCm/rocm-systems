#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
import glob
import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir",
        action="store",
        help="Path to output directory.",
    )


@pytest.fixture
def output_dir(request):
    return request.config.getoption("--output-dir")


@pytest.fixture
def pass1_agent_info(output_dir):
    """Agent info from pass 1"""
    filename_pattern = os.path.join(output_dir, "pass_1", "out_*_agent_info.csv")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def pass1_counter_data(output_dir):
    """Counter data from pass 1"""
    filename_pattern = os.path.join(output_dir, "pass_1", "out_*_counter_collection.csv")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def pass2_agent_info(output_dir):
    """Agent info from pass 2"""
    filename_pattern = os.path.join(output_dir, "pass_2", "out_*_agent_info.csv")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def pass2_counter_data(output_dir):
    """Counter data from pass 2"""
    filename_pattern = os.path.join(output_dir, "pass_2", "out_*_counter_collection.csv")
    matches = glob.glob(filename_pattern)
    assert (
        len(matches) == 1
    ), f"Expected 1 file matching {filename_pattern}, found {len(matches)}"
    data = []
    with open(matches[0], "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data
