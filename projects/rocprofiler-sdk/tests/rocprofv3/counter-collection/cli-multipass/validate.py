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

import os
import sys
import pytest


def test_pass_directories_exist(output_dir):
    """Verify that pass_1 and pass_2 directories were created"""
    pass1_dir = os.path.join(output_dir, "pass_1")
    pass2_dir = os.path.join(output_dir, "pass_2")

    assert os.path.isdir(pass1_dir), f"Expected pass_1 directory at {pass1_dir}"
    assert os.path.isdir(pass2_dir), f"Expected pass_2 directory at {pass2_dir}"


def test_pass1_agent_info(pass1_agent_info):
    """Validate agent info from pass 1"""
    assert len(pass1_agent_info) > 0, "No agent info found in pass 1"

    for row in pass1_agent_info:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0


def test_pass2_agent_info(pass2_agent_info):
    """Validate agent info from pass 2"""
    assert len(pass2_agent_info) > 0, "No agent info found in pass 2"

    for row in pass2_agent_info:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0


def test_pass1_counters(pass1_counter_data):
    """Validate counters from pass 1 (SQ_WAVES)"""
    assert len(pass1_counter_data) > 0, "No counter data found in pass 1"

    # All counters in pass 1 should be SQ_WAVES
    for row in pass1_counter_data:
        assert (
            row["Counter_Name"] == "SQ_WAVES"
        ), f"Expected SQ_WAVES in pass 1, got {row['Counter_Name']}"
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0
        assert len(row["Counter_Value"]) > 0
        # SQ_WAVES should have positive values for kernels
        assert float(row["Counter_Value"]) >= 0


def test_pass2_counters(pass2_counter_data):
    """Validate counters from pass 2 (GRBM_COUNT)"""
    assert len(pass2_counter_data) > 0, "No counter data found in pass 2"

    # All counters in pass 2 should be GRBM_COUNT
    for row in pass2_counter_data:
        assert (
            row["Counter_Name"] == "GRBM_COUNT"
        ), f"Expected GRBM_COUNT in pass 2, got {row['Counter_Name']}"
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0
        assert len(row["Counter_Value"]) > 0
        # GRBM_COUNT should have positive values
        assert float(row["Counter_Value"]) >= 0


def test_same_kernel_count_both_passes(pass1_counter_data, pass2_counter_data):
    """Verify that both passes collected data for the same number of kernel dispatches"""
    # Get unique dispatch IDs from both passes
    pass1_dispatch_ids = set([int(row["Dispatch_Id"]) for row in pass1_counter_data])
    pass2_dispatch_ids = set([int(row["Dispatch_Id"]) for row in pass2_counter_data])

    # Both passes should have collected data for the same dispatches
    assert (
        pass1_dispatch_ids == pass2_dispatch_ids
    ), f"Pass 1 and Pass 2 have different dispatch IDs. Pass1: {sorted(pass1_dispatch_ids)}, Pass2: {sorted(pass2_dispatch_ids)}"


def test_counter_separation(pass1_counter_data, pass2_counter_data):
    """Verify that counters are properly separated between passes"""
    # Get all counter names from both passes
    pass1_counters = set([row["Counter_Name"] for row in pass1_counter_data])
    pass2_counters = set([row["Counter_Name"] for row in pass2_counter_data])

    # Verify no overlap (each counter should only be in one pass)
    overlap = pass1_counters & pass2_counters
    assert (
        len(overlap) == 0
    ), f"Counters should not overlap between passes. Found {overlap} in both passes"

    # Verify we have the expected counters
    assert pass1_counters == {
        "SQ_WAVES"
    }, f"Expected only SQ_WAVES in pass 1, got {pass1_counters}"
    assert pass2_counters == {
        "GRBM_COUNT"
    }, f"Expected only GRBM_COUNT in pass 2, got {pass2_counters}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
