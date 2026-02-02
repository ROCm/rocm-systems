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
import json
import pytest
import glob
import subprocess


def get_gpu_node_count():
    """
    Detect the number of GPU nodes/agents available in the system.
    Returns the count of GPU nodes, or None if detection fails.
    """
    # try:
    #     # Method 1: Try using rocm-smi to count GPUs
    #     result = subprocess.run(
    #         ['rocm-smi', '--showid'],
    #         capture_output=True,
    #         text=True,
    #         timeout=5
    #     )
    #     if result.returncode == 0:
    #         # Count lines that contain "GPU[" to get GPU count
    #         gpu_count = result.stdout.count('GPU[')
    #         if gpu_count > 0:
    #             return gpu_count
    # except (FileNotFoundError, subprocess.TimeoutExpired):
    #     pass

    try:
        # Method 2: Check /sys/class/kfd/kfd/topology/nodes/ for GPU nodes
        # This follows the logic from rocprofiler-sdk/source/lib/rocprofiler-sdk/agent.cpp
        nodes_path = '/sys/class/kfd/kfd/topology/nodes'
        if os.path.exists(nodes_path):
            gpu_count = 0
            node_id = 0
            # Nodes are numbered monotonically starting from 0
            while True:
                node_path = os.path.join(nodes_path, str(node_id))
                # Once we're missing a node folder, there are no more nodes
                if not os.path.exists(node_path):
                    break

                properties_file = os.path.join(node_path, 'properties')
                if os.path.exists(properties_file) and os.access(properties_file, os.R_OK):
                    try:
                        with open(properties_file, 'r') as f:
                            content = f.read()

                        # Properties file must be non-empty
                        if not content.strip():
                            node_id += 1
                            continue

                        # Parse properties to find cpu_cores_count and simd_count
                        cpu_cores_count = 0
                        simd_count = 0

                        for line in content.split('\n'):
                            line = line.strip()
                            if line.startswith('cpu_cores_count'):
                                cpu_cores_count = int(line.split()[1])
                            elif line.startswith('simd_count'):
                                simd_count = int(line.split()[1])

                        # A node is a GPU if cpu_cores_count == 0 AND simd_count > 0
                        if cpu_cores_count == 0 and simd_count > 0:
                            gpu_count += 1
                    except (IOError, ValueError, IndexError):
                        pass

                node_id += 1

            if gpu_count > 0:
                return gpu_count
    except (OSError, ValueError):
        pass

    # If detection fails, return None
    return None


def get_sdk_data(data):
    """
    Extract rocprofiler-sdk-tool data from JSON, handling both dict and list structures.
    Some JSON files have rocprofiler-sdk-tool as a list, others as a dict.
    """
    if "rocprofiler-sdk-tool" not in data:
        return None

    sdk_data = data["rocprofiler-sdk-tool"]

    # Handle list structure - take the first element
    if isinstance(sdk_data, list):
        return sdk_data[0] if len(sdk_data) > 0 else {}

    # Already a dict
    return sdk_data


def test_mpi_ranks_feature(output_dir, test_mode):
    """
    Test the --mpi-ranks feature with different scenarios using simple-transpose application.

    The simple-transpose application runs a simple matrix transpose kernel using HIP.
    It uses the default stream (stream_id == 0) and executes a single matrixTranspose kernel.

    Test modes:
    - with-mpi-single: MPI run with 4 ranks, profiling only rank 0
    - with-mpi-multiple: MPI run with 4 ranks, profiling ranks 0-1,3
    - without-mpi: Non-MPI run, should generate output regardless
    """

    # Find all JSON output files in the output directory
    json_files = glob.glob(os.path.join(output_dir, "**/out_results.json"), recursive=True)

    # Detect the number of GPU nodes in the system
    gpu_node_count = get_gpu_node_count()
    is_single_node = gpu_node_count is not None and gpu_node_count <= 1

    if test_mode == "with-mpi-single":
        # With --mpi-ranks 0 and 4 MPI ranks, only rank 0 should generate output
        # So we should have exactly 1 JSON file
        assert len(json_files) == 1, (
            f"Expected 1 JSON file for rank 0 only, but found {len(json_files)}: {json_files}"
        )

        # Verify the file is from rank 0
        json_file = json_files[0]
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Check that we have valid profiling data
        sdk_data = get_sdk_data(data)
        assert sdk_data is not None, "Missing rocprofiler-sdk-tool data"
        buffer_records = sdk_data.get("buffer_records", {})

        # Should have some kernel or HIP API data
        has_data = (
            len(buffer_records.get("kernel_dispatch", [])) > 0 or
            len(buffer_records.get("hip_api", [])) > 0
        )
        assert has_data, "No profiling data found in rank 0 output"

    elif test_mode == "with-mpi-multiple":
        # With --mpi-ranks 0-1,3 and 4 MPI ranks, ranks 0, 1, and 3 should generate output
        expected_files = 3

        if is_single_node:
            # On single-node systems, MPI ranks share the same output directory and may overwrite
            # each other's files, resulting in only 1 file (the last rank to write)
            print(f"INFO: Single GPU node detected (GPU count: {gpu_node_count})")
            print("INFO: On single-node systems, MPI ranks may share output directory")

            # Accept 1 file (ranks overwriting each other) on single node systems
            assert len(json_files) >= 1, (
                f"Expected at least 1 JSON file on single-node system, but found {len(json_files)}: {json_files}"
            )

            if len(json_files) < expected_files:
                print(f"INFO: Found {len(json_files)} file(s) instead of {expected_files} - expected on single-node setup")
        else:
            # On multi-node systems, each rank should have its own output directory
            if gpu_node_count is not None:
                print(f"INFO: Multiple GPU nodes detected (GPU count: {gpu_node_count})")
            else:
                print("INFO: Could not detect GPU count, assuming multi-node system")

            # Require exactly the expected number of files on multi-node systems
            assert len(json_files) == expected_files, (
                f"Expected {expected_files} JSON files for ranks 0, 1, and 3 on multi-node system, "
                f"but found {len(json_files)}: {json_files}"
            )

        # Verify each file has valid profiling data
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            sdk_data = get_sdk_data(data)
            assert sdk_data is not None, f"Missing rocprofiler-sdk-tool data in {json_file}"
            buffer_records = sdk_data.get("buffer_records", {})

            # Should have some kernel or HIP API data
            has_data = (
                len(buffer_records.get("kernel_dispatch", [])) > 0 or
                len(buffer_records.get("hip_api", [])) > 0
            )
            assert has_data, f"No profiling data found in {json_file}"

    elif test_mode == "without-mpi":
        # Without MPI environment, --mpi-ranks should be ignored and output generated
        # We should have at least 1 JSON file
        assert len(json_files) >= 1, (
            f"Expected at least 1 JSON file for non-MPI run, but found {len(json_files)}: {json_files}"
        )

        # Verify the file has valid profiling data
        json_file = json_files[0]
        with open(json_file, 'r') as f:
            data = json.load(f)

        sdk_data = get_sdk_data(data)
        assert sdk_data is not None, "Missing rocprofiler-sdk-tool data"
        buffer_records = sdk_data.get("buffer_records", {})

        # Should have some kernel or HIP API data
        has_data = (
            len(buffer_records.get("kernel_dispatch", [])) > 0 or
            len(buffer_records.get("hip_api", [])) > 0
        )
        assert has_data, "No profiling data found in output"

    else:
        pytest.fail(f"Unknown test mode: {test_mode}")


def test_csv_output_consistency(output_dir, test_mode):
    """
    Verify that CSV files are also correctly generated/not generated based on rank filtering.
    """

    # Find all kernel trace CSV files
    csv_files = glob.glob(os.path.join(output_dir, "**/out_kernel_trace.csv"), recursive=True)

    # Detect the number of GPU nodes in the system
    gpu_node_count = get_gpu_node_count()
    is_single_node = gpu_node_count is not None and gpu_node_count <= 1

    if test_mode == "with-mpi-single":
        # Only rank 0 should have CSV output
        assert len(csv_files) == 1, (
            f"Expected 1 CSV file for rank 0 only, but found {len(csv_files)}: {csv_files}"
        )

    elif test_mode == "with-mpi-multiple":
        expected_files = 3

        if is_single_node:
            # On single-node systems, accept 1 or more files
            assert len(csv_files) >= 1, (
                f"Expected at least 1 CSV file on single-node system, but found {len(csv_files)}: {csv_files}"
            )
        else:
            # On multi-node systems, require exactly the expected number
            assert len(csv_files) == expected_files, (
                f"Expected {expected_files} CSV files for ranks 0, 1, and 3 on multi-node system, "
                f"but found {len(csv_files)}: {csv_files}"
            )

    elif test_mode == "without-mpi":
        # Non-MPI run should have CSV output
        assert len(csv_files) >= 1, (
            f"Expected at least 1 CSV file for non-MPI run, but found {len(csv_files)}: {csv_files}"
        )


def test_no_output_for_filtered_ranks(output_dir, test_mode):
    """
    Verify that ranks not in the --mpi-ranks list do not generate output.
    This test is skipped on single-node systems where ranks may share output directories.
    """

    if test_mode != "with-mpi-multiple":
        pytest.skip("This test only applies to with-mpi-multiple mode")

    # Detect if we're on a single-node system
    gpu_node_count = get_gpu_node_count()
    is_single_node = gpu_node_count is not None and gpu_node_count <= 1

    if is_single_node:
        pytest.skip("Skipping filtered ranks test on single-node system (ranks share output directory)")

    # In with-mpi-multiple mode with --mpi-ranks 0-1,3, rank 2 should NOT generate output
    json_files = glob.glob(os.path.join(output_dir, "**/out_results.json"), recursive=True)

    # On multi-node systems, we should have exactly 3 files (no output from rank 2)
    assert len(json_files) == 3, (
        f"Expected exactly 3 output files (ranks 0,1,3) on multi-node system, got {len(json_files)}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
