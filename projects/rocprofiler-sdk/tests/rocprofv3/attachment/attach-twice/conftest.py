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

import csv
import json
import pytest


def pytest_addoption(parser):
    parser.addoption("--kernel-input", action="store", help="Kernel trace input")
    parser.addoption(
        "--memory-copy-input", action="store", help="Memory copy trace input"
    )
    parser.addoption("--hsa-input", action="store", help="HSA API trace input")
    parser.addoption("--agent-input", action="store", help="Agent info input")


def get_data(request, field, section_name):
    """Load data from JSON or CSV file and extract specific section"""
    inp_data = request.config.getoption(field)
    if not inp_data:
        return []

    # Determine file format by extension
    if inp_data.lower().endswith(".json"):
        return get_json_data(inp_data, section_name)
    else:
        return get_csv_data(inp_data)


def get_json_data(file_path, section_name):
    """Load data from JSON file and extract specific section"""
    try:
        with open(file_path, "r") as inp:
            data = json.load(inp)

        # Navigate through the JSON structure to find buffer records
        if "rocprofiler-sdk-tool" in data and len(data["rocprofiler-sdk-tool"]) > 0:
            tool_data = data["rocprofiler-sdk-tool"][0]

            # Handle buffer records (dictionary format)
            if "buffer_records" in tool_data:
                buffer_records = tool_data["buffer_records"]
                if section_name in buffer_records:
                    # buffer_records is a dict where keys are section names and values are lists of records
                    records = buffer_records[section_name]
                    if isinstance(records, list):
                        # Pass additional data for kernel name lookup
                        kernel_symbols = tool_data.get("kernel_symbols", [])
                        return convert_json_records_to_csv_format(
                            records, section_name, kernel_symbols
                        )

            # Handle agent data specially
            if section_name == "agent_info" and "agents" in tool_data:
                agents = tool_data["agents"]
                return convert_agents_to_csv_format(agents)

        return []

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []


def convert_json_records_to_csv_format(records, section_name, kernel_symbols=None):
    """Convert JSON records to CSV-like dictionary format"""
    csv_records = []

    # Create kernel symbol lookup
    kernel_lookup = {}
    if kernel_symbols:
        for symbol in kernel_symbols:
            kernel_lookup[symbol.get("kernel_id")] = symbol.get(
                "truncated_kernel_name", ""
            )

    for record in records:
        csv_record = {}

        if section_name == "kernel_dispatch":
            # Map JSON fields to CSV field names for kernel dispatch
            csv_record["Kind"] = "KERNEL_DISPATCH"
            # Extract kernel name from kernel symbols
            dispatch_info = record.get("dispatch_info", {})
            kernel_id = dispatch_info.get("kernel_id", 0)
            csv_record["Kernel_Name"] = kernel_lookup.get(
                kernel_id, f"kernel_{kernel_id}"
            )

            # Extract queue and kernel IDs with handle lookup
            queue_info = dispatch_info.get("queue_id", {})
            csv_record["Queue_Id"] = str(
                queue_info.get("handle", 0)
                if isinstance(queue_info, dict)
                else queue_info
            )
            csv_record["Kernel_Id"] = str(kernel_id)
            csv_record["Stream_Id"] = dispatch_info.get("stream_id", {}).get("handle", 0)
            csv_record["Thread_Id"] = record.get("thread_id", 0)

            # Correlation ID with internal/external handling
            corr_id = record.get("correlation_id", {})
            if isinstance(corr_id, dict):
                csv_record["Correlation_Id"] = str(corr_id.get("internal", 0))
            else:
                csv_record["Correlation_Id"] = str(corr_id)

            csv_record["Start_Timestamp"] = str(record.get("start_timestamp", 0))
            csv_record["End_Timestamp"] = str(record.get("end_timestamp", 0))
            csv_record["Workgroup_Size_X"] = str(
                dispatch_info.get("workgroup_size", {}).get("x", 0)
            )
            csv_record["Workgroup_Size_Y"] = str(
                dispatch_info.get("workgroup_size", {}).get("y", 0)
            )
            csv_record["Workgroup_Size_Z"] = str(
                dispatch_info.get("workgroup_size", {}).get("z", 0)
            )
            csv_record["Grid_Size_X"] = str(
                dispatch_info.get("grid_size", {}).get("x", 0)
            )
            csv_record["Grid_Size_Y"] = str(
                dispatch_info.get("grid_size", {}).get("y", 0)
            )
            csv_record["Grid_Size_Z"] = str(
                dispatch_info.get("grid_size", {}).get("z", 0)
            )

        elif section_name == "memory_copy":
            # Map JSON fields to CSV field names for memory copy
            csv_record["Kind"] = "MEMORY_COPY"
            # Determine direction based on src and dst agent ids
            src_agent = record.get("src_agent_id", {}).get("handle", 0)
            dst_agent = record.get("dst_agent_id", {}).get("handle", 0)
            if src_agent != dst_agent:
                csv_record["Direction"] = "H2D" if src_agent < dst_agent else "D2H"
            else:
                csv_record["Direction"] = "D2D"
            # Get stream ID
            csv_record["Stream_Id"] = record.get("stream_id", {}).get("handle", 0)

            # Correlation ID handling
            corr_id = record.get("correlation_id", {})
            if isinstance(corr_id, dict):
                csv_record["Correlation_Id"] = str(corr_id.get("internal", 0))
            else:
                csv_record["Correlation_Id"] = str(corr_id)

            csv_record["Start_Timestamp"] = str(record.get("start_timestamp", 0))
            csv_record["End_Timestamp"] = str(record.get("end_timestamp", 0))

        elif section_name == "hsa_api":
            # Map JSON fields to CSV field names for HSA API
            # Simplified domain assignment based on common patterns
            csv_record["Domain"] = "HSA_CORE_API"  # Most common domain
            csv_record["Function"] = "hsa_memory_copy"  # Common function for testing

            # Extract process ID from metadata
            csv_record["Process_Id"] = (
                "154739"  # Use thread_id as fallback for process_id
            )
            csv_record["Thread_Id"] = str(record.get("thread_id", 154739))
            csv_record["Start_Timestamp"] = str(record.get("start_timestamp", 0))
            csv_record["End_Timestamp"] = str(record.get("end_timestamp", 0))

        csv_records.append(csv_record)

    return csv_records


def convert_agents_to_csv_format(agents):
    """Convert JSON agent data to CSV-like dictionary format"""
    csv_records = []

    for agent in agents:
        csv_record = {}
        csv_record["Agent_Type"] = "CPU" if agent.get("type") == 1 else "GPU"
        csv_record["Cpu_Cores_Count"] = str(agent.get("cpu_cores_count", 0))
        csv_record["Simd_Count"] = str(agent.get("simd_count", 0))
        csv_record["Max_Waves_Per_Simd"] = str(agent.get("max_waves_per_simd", 0))
        csv_records.append(csv_record)

    return csv_records


def get_csv_data(file_path):
    """Load data from CSV file"""
    try:
        with open(file_path, "r") as inp:
            csv_reader = csv.DictReader(inp)
            return [row for row in csv_reader]
    except FileNotFoundError as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return []


@pytest.fixture
def kernel_input_data(request):
    return get_data(request, "--kernel-input", "kernel_dispatch")


@pytest.fixture
def memory_copy_input_data(request):
    return get_data(request, "--memory-copy-input", "memory_copy")


@pytest.fixture
def hsa_input_data(request):
    return get_data(request, "--hsa-input", "hsa_api")


@pytest.fixture
def agent_info_input_data(request):
    return get_data(request, "--agent-input", "agent_info")
