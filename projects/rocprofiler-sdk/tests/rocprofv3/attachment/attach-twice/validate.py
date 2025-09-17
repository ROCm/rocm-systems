#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
import csv
import json
import os


def test_attachment_kernel_trace(kernel_input_data):
    """Verify that kernel traces were captured during attachment."""

    # We should have captured some kernel dispatches
    assert len(kernel_input_data) > 0, "No kernel dispatches captured during attachment"

    # The test app launches a kernel called "simple_kernel"
    kernel_names = [row["Kernel_Name"] for row in kernel_input_data]

    # Check that we captured the simple_kernel
    simple_kernel_found = any("simple_kernel" in name for name in kernel_names)
    assert (
        simple_kernel_found
    ), f"Expected 'simple_kernel' not found in kernel names: {kernel_names}"

    # Verify basic kernel properties
    for row in kernel_input_data:
        if "simple_kernel" in row["Kernel_Name"]:
            assert row["Kind"] == "KERNEL_DISPATCH"
            assert int(row["Queue_Id"]) > 0
            assert int(row["Kernel_Id"]) > 0
            assert int(row["Correlation_Id"]) > 0
            assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

            # Verify kernel dimensions (from the test app)
            assert int(row["Workgroup_Size_X"]) == 256  # threads_per_block
            assert int(row["Workgroup_Size_Y"]) == 1
            assert int(row["Workgroup_Size_Z"]) == 1

            assert int(row["Grid_Size_X"]) >= 1
            assert int(row["Grid_Size_Y"]) >= 1
            assert int(row["Grid_Size_Z"]) >= 1


def test_attachment_memory_copy_trace(memory_copy_input_data):
    """Verify that memory copy operations were captured during attachment."""

    # We should have captured memory copies (HtoD and DtoH)
    assert (
        len(memory_copy_input_data) > 0
    ), "No memory copy operations captured during attachment"

    host_to_device_count = 0
    device_to_host_count = 0

    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"
        assert int(row["Correlation_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

        # Count the direction of memory copies
        if "MEMORY_COPY_HOST_TO_DEVICE" in row["Direction"] or "H2D" in row["Direction"]:
            host_to_device_count += 1
        elif (
            "MEMORY_COPY_DEVICE_TO_HOST" in row["Direction"] or "D2H" in row["Direction"]
        ):
            device_to_host_count += 1

    # We should have both H2D and D2H copies
    assert host_to_device_count > 0, "No host-to-device memory copies captured"
    assert device_to_host_count > 0, "No device-to-host memory copies captured"


def test_attachment_hsa_api_trace(hsa_input_data):
    """Verify that HSA API calls were captured during attachment."""

    # Should have some HSA API calls
    assert len(hsa_input_data) > 0, "No HSA API calls captured during attachment"

    functions = []
    for row in hsa_input_data:
        assert row["Domain"] in (
            "HSA_CORE_API",
            "HSA_AMD_EXT_API",
            "HSA_IMAGE_EXT_API",
            "HSA_FINALIZE_EXT_API",
        )
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) >= int(row["Process_Id"])
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])

    # Note: Memory-related HSA function checking disabled for RHEL/SLES compatibility
    # assert any(
    #     "memory" in func.lower() for func in functions
    # ), "No memory-related HSA functions captured"


def test_agent_info(agent_info_input_data):
    """Verify agent information is captured correctly."""

    assert len(agent_info_input_data) > 0, "No agent information captured"

    cpu_count = 0
    gpu_count = 0

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")

        if agent_type == "CPU":
            cpu_count += 1
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            gpu_count += 1
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0

    # Should have at least one GPU for the test
    assert gpu_count > 0, "No GPU agents found"


if __name__ == "__main__":
    # For running directly without pytest
    import argparse

    parser = argparse.ArgumentParser(description="Validate attachment test output")

    # Support both individual inputs and single JSON input
    parser.add_argument("--json-input", help="Single JSON file containing all trace data")
    parser.add_argument("--kernel-input", help="Kernel trace CSV/JSON file")
    parser.add_argument("--memory-copy-input", help="Memory copy trace CSV/JSON file")
    parser.add_argument("--hsa-input", help="HSA API trace CSV/JSON file")
    parser.add_argument("--agent-input", help="Agent info CSV/JSON file")

    args = parser.parse_args()

    # Validate arguments
    if args.json_input:
        # Use the single JSON file for all inputs
        if not os.path.exists(args.json_input):
            print(f"Error: JSON input file {args.json_input} does not exist")
            sys.exit(1)
        args.kernel_input = args.json_input
        args.memory_copy_input = args.json_input
        args.hsa_input = args.json_input
        args.agent_input = args.json_input
    elif not all(
        [args.kernel_input, args.memory_copy_input, args.hsa_input, args.agent_input]
    ):
        print(
            "Error: Either --json-input or all four individual inputs (--kernel-input, --memory-copy-input, --hsa-input, --agent-input) are required"
        )
        sys.exit(1)

    # Load data files (CSV or JSON)
    def load_data_file(filepath, section_name):
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} does not exist")
            return []

        if filepath.lower().endswith(".json"):
            return load_json_data(filepath, section_name)
        else:
            return load_csv_data(filepath)

    def load_csv_data(filepath):
        with open(filepath, "r") as f:
            return list(csv.DictReader(f))

    def load_json_data(filepath, section_name):
        from conftest import get_json_data

        return get_json_data(filepath, section_name)

    kernel_data = load_data_file(args.kernel_input, "kernel_dispatch")
    memory_copy_data = load_data_file(args.memory_copy_input, "memory_copy")
    hsa_data = load_data_file(args.hsa_input, "hsa_api")
    agent_data = load_data_file(args.agent_input, "agent_info")

    # Run tests
    try:
        test_attachment_kernel_trace(kernel_data)
        print("[PASS] Kernel trace validation passed")

        test_attachment_memory_copy_trace(memory_copy_data)
        print("[PASS] Memory copy trace validation passed")

        test_attachment_hsa_api_trace(hsa_data)
        print("[PASS] HSA API trace validation passed")

        test_agent_info(agent_data)
        print("[PASS] Agent info validation passed")

        print("\nAll attachment tests passed!")
        sys.exit(0)

    except AssertionError as e:
        print(f"[FAIL] Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
