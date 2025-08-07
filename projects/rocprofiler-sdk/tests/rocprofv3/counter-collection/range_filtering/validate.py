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

import sys
import pytest
import numpy as np
import pandas as pd
import re


def unique(lst):
    return list(set(lst))


def validate_json(input_data):
    json_data = input_data["json_data"]
    iteration_list = input_data["iteration_range"]

    data = json_data["rocprofiler-sdk-tool"]
    counter_collection_data = data["callback_records"]["counter_collection"]
    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    dispatch_ids = {}

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    iteration = 1
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        kernel_name = get_kernel_name(dispatch_info["kernel_id"])
        if kernel_name == "transpose" and iteration in iteration_list:
            dispatch_ids[dispatch_info["dispatch_id"]] = dispatch_info
        if kernel_name == "transpose":
            iteration = iteration + 1

    for counter in counter_collection_data:
        dispatch_data = counter["dispatch_data"]["dispatch_info"]
        dispatch_id = dispatch_data["dispatch_id"]
        if dispatch_id in dispatch_ids.keys():
            assert dispatch_data == dispatch_ids[dispatch_id]


def test_validate_counter_collection_pass1(input_json_pass1):
    validate_json(input_json_pass1)


def test_validate_counter_collection_pass2(input_json_pass2):
    validate_json(input_json_pass2)


def test_validate_counter_collection_pass3(input_json_pass3):
    validate_json(input_json_pass3)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
