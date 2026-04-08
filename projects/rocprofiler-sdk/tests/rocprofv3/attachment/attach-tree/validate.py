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

    kernel_threads = set()
    kernel_streams = set()
    NUM_KERNEL_THREADS = 32
    expected_stream_ids = set([i for i in range(1, 9)])

    # Verify basic kernel properties
    for row in kernel_input_data:
        if "simple_kernel" in row["Kernel_Name"]:
            assert "Stream_Id" in row
            assert "Thread_Id" in row
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

            thread_id = int(row["Thread_Id"])
            stream_id = int(row["Stream_Id"])
            kernel_threads.add(thread_id)
            kernel_streams.add(stream_id)

    # Exactly 8 streams and 32 threads
    len(kernel_threads) == NUM_KERNEL_THREADS
    kernel_streams == expected_stream_ids


def test_attachment_memory_copy_trace(memory_copy_input_data):
    """Verify that memory copy operations were captured during attachment."""

    # We should have captured memory copies (HtoD and DtoH)
    assert (
        len(memory_copy_input_data) > 0
    ), "No memory copy operations captured during attachment"

    host_to_device_count = 0
    device_to_host_count = 0
    memory_copy_streams = set()
    expected_stream_ids = set([i for i in range(1, 9)])

    for row in memory_copy_input_data:
        assert "Stream_Id" in row
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

        stream_id = int(row["Stream_Id"])
        memory_copy_streams.add(stream_id)

    # We should have both H2D and D2H copies
    assert host_to_device_count > 0, "No host-to-device memory copies captured"
    assert device_to_host_count > 0, "No device-to-host memory copies captured"
    # Exactly 8 streams
    memory_copy_streams == expected_stream_ids


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
        assert int(row["Thread_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])

    assert any(
        "memory" in func.lower() for func in functions
    ), "No memory-related HSA functions captured"


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
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
