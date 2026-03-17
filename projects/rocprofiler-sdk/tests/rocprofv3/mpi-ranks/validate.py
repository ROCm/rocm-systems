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
    try:
        # Method 2: Check /sys/class/kfd/kfd/topology/nodes/ for GPU nodes
        # This follows the logic from rocprofiler-sdk/source/lib/rocprofiler-sdk/agent.cpp
        nodes_path = "/sys/class/kfd/kfd/topology/nodes"
        if os.path.exists(nodes_path):
            gpu_count = 0
            node_id = 0
            # Nodes are numbered monotonically starting from 0
            while True:
                node_path = os.path.join(nodes_path, str(node_id))
                # Once we're missing a node folder, there are no more nodes
                if not os.path.exists(node_path):
                    break

                properties_file = os.path.join(node_path, "properties")
                if os.path.exists(properties_file) and os.access(
                    properties_file, os.R_OK
                ):
                    try:
                        with open(properties_file, "r") as f:
                            content = f.read()

                        # Properties file must be non-empty
                        if not content.strip():
                            node_id += 1
                            continue

                        # Parse properties to find cpu_cores_count and simd_count
                        cpu_cores_count = 0
                        simd_count = 0

                        for line in content.split("\n"):
                            line = line.strip()
                            if line.startswith("cpu_cores_count"):
                                cpu_cores_count = int(line.split()[1])
                            elif line.startswith("simd_count"):
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


def load_json_file(filepath):
    """
    Load JSON from a file, handling cases where multiple MPI ranks may have written
    to the same file (resulting in concatenated JSON objects).

    Returns the first valid JSON object found, or None if no valid JSON exists.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Try to load as a single JSON object first
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # If that fails, it might be multiple JSON objects concatenated
            # Try to extract the first valid JSON object
            decoder = json.JSONDecoder()
            try:
                obj, idx = decoder.raw_decode(content)
                # Successfully decoded the first JSON object
                return obj
            except json.JSONDecodeError:
                # Even the first object is malformed
                return None
    except (IOError, OSError):
        return None


def get_sdk_data(data):
    """
    Extract rocprofiler-sdk-tool data from JSON, handling both dict and list structures.
    Some JSON files have rocprofiler-sdk-tool as a list, others as a dict.
    """
    if data is None or "rocprofiler-sdk-tool" not in data:
        return None

    sdk_data = data["rocprofiler-sdk-tool"]

    # Handle list structure - take the first element
    if isinstance(sdk_data, list):
        return sdk_data[0] if len(sdk_data) > 0 else {}

    # Already a dict
    return sdk_data


def validate_json_file_has_profiling_data(json_file):
    """
    Validate that a JSON file contains valid profiling data.

    Loads the JSON file, extracts rocprofiler-sdk-tool data, and verifies that
    it contains either kernel_dispatch or hip_api profiling data.

    Raises AssertionError if validation fails.
    """
    data = load_json_file(json_file)
    assert data is not None, f"Failed to load JSON from {json_file}"

    sdk_data = get_sdk_data(data)
    assert sdk_data is not None, f"Missing rocprofiler-sdk-tool data in {json_file}"
    buffer_records = sdk_data.get("buffer_records", {})

    # Should have some kernel or HIP API data
    has_data = (
        len(buffer_records.get("kernel_dispatch", [])) > 0
        or len(buffer_records.get("hip_api", [])) > 0
    )
    assert has_data, f"No profiling data found in {json_file}"


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

    # Detect the number of GPU nodes in the system
    gpu_node_count = get_gpu_node_count()
    is_single_node = gpu_node_count is not None and gpu_node_count <= 1

    # Find JSON output files
    # For MPI tests: Look only in rank.* subdirectories (to avoid stale files)
    # For non-MPI tests: Look everywhere in the output directory
    if test_mode in ["with-mpi-single", "with-mpi-multiple"]:
        # MPI test - only look in rank.* subdirectories
        json_files = []
        for rank_dir in glob.glob(os.path.join(output_dir, "rank.*")):
            json_files.extend(
                glob.glob(os.path.join(rank_dir, "**/out_results.json"), recursive=True)
            )
    else:
        # Non-MPI test - look everywhere
        json_files = glob.glob(
            os.path.join(output_dir, "**/out_results.json"), recursive=True
        )

    if test_mode == "with-mpi-single":
        # With --mpi-ranks 0 and 4 MPI ranks, only rank 0 should generate output in rank.0/
        # So we should have exactly 1 JSON file
        assert (
            len(json_files) == 1
        ), f"Expected 1 JSON file for rank 0 only, but found {len(json_files)}: {json_files}"

        # Verify the file is from rank 0
        validate_json_file_has_profiling_data(json_files[0])

    elif test_mode == "with-mpi-multiple":
        # With --mpi-ranks 0-1,3 and 4 MPI ranks, ranks 0, 1, and 3 should generate output
        # Each rank uses a separate output directory (via %env{OMPI_COMM_WORLD_RANK}%)
        # so we should always get exactly 3 files
        expected_files = 3

        if gpu_node_count is not None:
            print(f"INFO: Detected {gpu_node_count} GPU node(s)")

        # With separate output directories per rank, we should have exactly the expected number
        assert len(json_files) == expected_files, (
            f"Expected {expected_files} JSON files for ranks 0, 1, and 3, "
            f"but found {len(json_files)}: {json_files}"
        )

        # Verify each file has valid profiling data
        for json_file in json_files:
            validate_json_file_has_profiling_data(json_file)

    elif test_mode == "without-mpi":
        # Without MPI environment, --mpi-ranks should be ignored and output generated
        # We should have at least 1 JSON file
        assert (
            len(json_files) >= 1
        ), f"Expected at least 1 JSON file for non-MPI run, but found {len(json_files)}: {json_files}"

        # Verify the file has valid profiling data
        validate_json_file_has_profiling_data(json_files[0])

    else:
        pytest.fail(f"Unknown test mode: {test_mode}")


def test_csv_output_consistency(output_dir, test_mode):
    """
    Verify that CSV files are also correctly generated/not generated based on rank filtering.
    """

    # Find CSV files
    # For MPI tests: Look only in rank.* subdirectories
    # For non-MPI tests: Look everywhere in the output directory
    if test_mode in ["with-mpi-single", "with-mpi-multiple"]:
        # MPI test - only look in rank.* subdirectories
        csv_files = []
        for rank_dir in glob.glob(os.path.join(output_dir, "rank.*")):
            csv_files.extend(
                glob.glob(
                    os.path.join(rank_dir, "**/out_kernel_trace.csv"), recursive=True
                )
            )
    else:
        # Non-MPI test - look everywhere
        csv_files = glob.glob(
            os.path.join(output_dir, "**/out_kernel_trace.csv"), recursive=True
        )

    if test_mode == "with-mpi-single":
        # Only rank 0 should have CSV output
        assert (
            len(csv_files) == 1
        ), f"Expected 1 CSV file for rank 0 only, but found {len(csv_files)}: {csv_files}"

    elif test_mode == "with-mpi-multiple":
        # Each rank has separate output directory, so expect exactly 3 CSV files
        expected_files = 3
        assert len(csv_files) == expected_files, (
            f"Expected {expected_files} CSV files for ranks 0, 1, and 3, "
            f"but found {len(csv_files)}: {csv_files}"
        )

    elif test_mode == "without-mpi":
        # Non-MPI run should have CSV output
        assert (
            len(csv_files) >= 1
        ), f"Expected at least 1 CSV file for non-MPI run, but found {len(csv_files)}: {csv_files}"


def test_no_output_for_filtered_ranks(output_dir, test_mode):
    """
    Verify that ranks not in the --mpi-ranks list do not generate output.
    Since each rank has a separate output directory (via %env{OMPI_COMM_WORLD_RANK}%),
    we can check that rank 2's directory doesn't exist or is empty.
    """

    if test_mode != "with-mpi-multiple":
        pytest.skip("This test only applies to with-mpi-multiple mode")

    # Check for rank.2 directory (which should NOT have been created or should be empty)
    rank_2_dir = os.path.join(output_dir, "rank.2")

    if os.path.exists(rank_2_dir):
        # Directory exists - check if it has any JSON files
        json_files_in_rank_2 = glob.glob(
            os.path.join(rank_2_dir, "**/out_results.json"), recursive=True
        )
        assert (
            len(json_files_in_rank_2) == 0
        ), f"Rank 2 should not generate output, but found {len(json_files_in_rank_2)} files: {json_files_in_rank_2}"

    # Verify that ranks 0, 1, and 3 directories exist with output
    for rank in [0, 1, 3]:
        rank_dir = os.path.join(output_dir, f"rank.{rank}")
        assert os.path.exists(
            rank_dir
        ), f"Expected directory for rank {rank} at {rank_dir}"
        json_files = glob.glob(
            os.path.join(rank_dir, "**/out_results.json"), recursive=True
        )
        assert (
            len(json_files) >= 1
        ), f"Expected output files for rank {rank} in {rank_dir}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
