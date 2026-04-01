#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import sys
from pathlib import Path


def print_help():
    """Print the help message"""
    print(f"""
    Unified Memory Output Validation Tool

    DESCRIPTION:
        This tool validates unified memory profiling output files (text and JSON formats).
        It checks for required fields, proper structure, and expected content.

    USAGE:
        {os.path.basename(__file__)} --output-dir <path> [OPTIONS]

    REQUIRED ARGUMENTS:
        --output-dir PATH           Directory containing unified_memory.txt and unified_memory.json

    OPTIONAL ARGUMENTS:
        --txt-file PATH             Explicit path to unified_memory.txt (overrides --output-dir)
        --json-file PATH            Explicit path to unified_memory.json (overrides --output-dir)
        -h, --help                  Show this help message and exit

    EXAMPLES:
        # Validate outputs in current directory
        {os.path.basename(__file__)} --output-dir .

        # Validate outputs in specific directory
        {os.path.basename(__file__)} --output-dir rocprof-sys-output

        # Validate specific files
        {os.path.basename(__file__)} --txt-file unified_memory.txt --json-file unified_memory.json

    VALIDATION CHECKS:
        - Text output format and required headers
        - JSON structure and required fields
        - Device information completeness
        - Migration statistics presence

    EXIT CODES:
        0  - All validations passed successfully
        1  - File not found or general error
        65 - Validation failures detected (EX_DATAERR)
    """)


def validate_text_output(filepath):
    """
    Validates the text output file for unified memory profiling.
    Checks for required headers and migration direction data.

    Args:
        filepath: Path object pointing to the unified_memory.txt file

    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"Validating text output: {filepath}")

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False

    content = filepath.read_text()

    required_headers = [
        "Unified Memory profiling result",
        "Device",
        "Count",
        "Avg Size",
        "Total Size",
        "Bandwidth",
        "Total Page Faults",
    ]

    missing = []
    for header in required_headers:
        if header not in content:
            missing.append(header)

    if missing:
        print(f"Error: Missing required headers in text output: {missing}")
        return False

    migration_directions = ["Host To Device", "Device To Host", "Device To Device"]

    has_migration = any(direction in content for direction in migration_directions)

    if not has_migration:
        print("Error: No migration statistics found in text output")
        return False

    print("Text output validation passed")
    return True


def validate_json_output(filepath):
    """
    Validates the JSON output file for unified memory profiling.
    Checks for proper structure, required fields, and device information.

    Args:
        filepath: Path object pointing to the unified_memory.json file

    Returns:
        bool: True if validation passes, False otherwise
    """
    print(f"Validating JSON output: {filepath}")

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return False

    if "devices" not in data:
        print("Error: Missing 'devices' array in JSON output")
        return False

    if "summary" not in data:
        print("Error: Missing 'summary' object in JSON output")
        return False

    summary = data["summary"]
    required_summary_fields = [
        "xnack_enabled",
        "total_page_faults",
        "migration_triggers",
    ]
    missing_fields = [field for field in required_summary_fields if field not in summary]

    if missing_fields:
        print(f"Error: Missing summary fields: {missing_fields}")
        return False

    devices = data["devices"]
    if not isinstance(devices, list):
        print("Error: 'devices' must be an array")
        return False

    if len(devices) == 0:
        print("Warning: No devices in output (may be expected if no migrations occurred)")

    for i, device in enumerate(devices):
        required_device_fields = ["device_id", "device_name", "migrations"]
        missing = [field for field in required_device_fields if field not in device]

        if missing:
            print(f"Error: Device {i} missing fields: {missing}")
            return False

        migrations = device["migrations"]
        required_directions = ["host_to_device", "device_to_host"]

        for direction in required_directions:
            if direction not in migrations:
                print(f"Error: Device {i} missing migration direction: {direction}")
                return False

            stats = migrations[direction]
            required_stats = [
                "count",
                "total_size_bytes",
                "min_size_bytes",
                "max_size_bytes",
                "avg_size_bytes",
                "total_time_ns",
                "bandwidth_gbps",
            ]
            missing_stats = [field for field in required_stats if field not in stats]

            if missing_stats:
                print(f"Error: Device {i}, {direction} missing stats: {missing_stats}")
                return False

    triggers = summary["migration_triggers"]
    required_trigger_fields = [
        "gpu_page_fault",
        "cpu_page_fault",
        "prefetch",
        "ttm_eviction",
        "unknown",
    ]
    missing_trigger_fields = [
        field for field in required_trigger_fields if field not in triggers
    ]
    if missing_trigger_fields:
        print(f"Error: Missing migration_triggers fields: {missing_trigger_fields}")
        return False

    print("JSON output validation passed")
    print(f"  Devices: {len(devices)}")
    print(f"  XNACK enabled: {summary['xnack_enabled']}")
    print(f"  Total page faults: {summary['total_page_faults']}")
    print(f"  Migration triggers: {triggers}")

    return True


def find_output_files(base_dir):
    """
    Locates unified_memory.txt and unified_memory.json in the output directory.
    Searches recursively to handle timestamped subdirectories.

    Args:
        base_dir: Base directory to search for output files

    Returns:
        tuple: (txt_file_path, json_file_path) or (None, None) if not found
    """
    base_path = Path(base_dir)

    txt_file = None
    json_file = None

    # First try direct match, then recursive search
    txt_patterns = ["unified_memory.txt", "**/unified_memory.txt"]
    json_patterns = ["unified_memory.json", "**/unified_memory.json"]

    for pattern in txt_patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            # If multiple matches, prefer the most recent (last modified)
            txt_file = max(matches, key=lambda p: p.stat().st_mtime)
            break

    for pattern in json_patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            # If multiple matches, prefer the most recent (last modified)
            json_file = max(matches, key=lambda p: p.stat().st_mtime)
            break

    return txt_file, json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory containing unified_memory.txt and unified_memory.json",
        default=None,
    )

    parser.add_argument(
        "--txt-file",
        type=Path,
        help="Explicit path to unified_memory.txt (overrides --output-dir)",
    )

    parser.add_argument(
        "--json-file",
        type=Path,
        help="Explicit path to unified_memory.json (overrides --output-dir)",
    )

    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(os.EX_OK)

    if args.txt_file and args.json_file:
        txt_file = args.txt_file
        json_file = args.json_file
    elif args.output_dir:
        txt_file, json_file = find_output_files(args.output_dir)
    else:
        print(
            "Error: Either --output-dir or both --txt-file and --json-file must be provided"
        )
        print_help()
        sys.exit(os.EX_USAGE)

    if not txt_file:
        print("Error: Could not find unified_memory.txt")
        sys.exit(1)

    if not json_file:
        print("Error: Could not find unified_memory.json")
        sys.exit(1)

    print(
        f"Validating unified memory output. Output directory: {args.output_dir or 'N/A'}"
    )
    print(f"Found unified memory outputs:")
    print(f"  Text:  {txt_file}")
    print(f"  JSON:  {json_file}")
    print()

    print("Starting unified memory output validation...")
    txt_valid = validate_text_output(txt_file)
    json_valid = validate_json_output(json_file)

    print()
    if txt_valid and json_valid:
        print("All validation checks passed")
        print(f"{txt_file} and {json_file} validated")
        sys.exit(os.EX_OK)
    else:
        print("Some validation checks failed")
        print(f"Failure validating unified memory outputs")
        sys.exit(os.EX_DATAERR)
