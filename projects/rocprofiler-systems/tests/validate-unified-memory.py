#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# BSD sysexits.h codes — portable across platforms (os.EX_* is POSIX-only)
EXIT_OK = 0
EXIT_USAGE = 64
EXIT_DATAERR = 65
EXIT_NOINPUT = 66


def print_help():
    """Print the help message"""
    print(f"""
    Unified Memory Output Validation Tool

    DESCRIPTION:
        This tool validates unified memory profiling output files (text and JSON formats).
        It checks for required fields, proper structure, and expected content.

    USAGE:
        {os.path.basename(__file__)} --txt-file <path> --json-file <path>
        {os.path.basename(__file__)} --output-dir <dir>

    ARGUMENT MODES (mutually exclusive):
        --txt-file PATH --json-file PATH    Explicit file paths
        --output-dir DIR                    Single-level lookup of unified_memory*.{{txt,json}}

    OPTIONAL ARGUMENTS:
        -h, --help                          Show this help message and exit

    EXAMPLES:
        {os.path.basename(__file__)} --txt-file unified_memory.txt --json-file unified_memory.json
        {os.path.basename(__file__)} --output-dir rocprof-sys-tests-output/unified-memory

    VALIDATION CHECKS:
        - Text output format and required headers
        - JSON structure and required fields
        - Device information completeness
        - Migration statistics presence

    EXIT CODES:
        0  - All validations passed successfully
        64 - Usage error (EX_USAGE)
        65 - Validation failures detected (EX_DATAERR)
        66 - Required input file not found (EX_NOINPUT)
    """)


def validate_text_output(filepath: Path) -> bool:
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

    base_headers = ["Unified Memory profiling result", "Total Page Faults"]
    missing = [h for h in base_headers if h not in content]
    if missing:
        print(f"Error: Missing required headers in text output: {missing}")
        return False

    has_device_block = 'Device "' in content

    if has_device_block:
        migration_headers = [
            "Count",
            "Avg Size",
            "Total Size",
            "Migration Throughput",
        ]
        missing = [h for h in migration_headers if h not in content]
        if missing:
            print(f"Error: Missing migration headers in text output: {missing}")
            return False

        migration_directions = ["Host To Device", "Device To Host", "Device To Device"]
        if not any(direction in content for direction in migration_directions):
            print("Error: Device block present but no migration statistics found")
            return False
    else:
        match = re.search(r"Total Page Faults:\s*(\d+)", content)
        if not match or int(match.group(1)) == 0:
            print(
                "Error: No migrations and no page faults captured; report should be empty"
            )
            return False

    print("Text output validation passed")
    return True


def validate_json_output(filepath: Path) -> bool:
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
        required_directions = ["host_to_device", "device_to_host", "device_to_device"]

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
                "migration_throughput_gbps",
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


def resolve_from_dir(output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Resolve unified_memory*.txt and unified_memory*.json in output_dir.
    Single-level lookup only (no recursion). Used for tests where the PID-suffixed
    filename is unknown at configure time.
    """
    txt_matches = sorted(output_dir.glob("unified_memory*.txt"))
    json_matches = sorted(output_dir.glob("unified_memory*.json"))

    txt_file = txt_matches[0] if txt_matches else None
    json_file = json_matches[0] if json_matches else None

    if len(txt_matches) > 1:
        print(
            f"Warning: multiple unified_memory*.txt files in {output_dir}; using {txt_file}"
        )
    if len(json_matches) > 1:
        print(
            f"Warning: multiple unified_memory*.json files in {output_dir}; using {json_file}"
        )

    return txt_file, json_file


if __name__ == "__main__":
    # Handle help manually so that --help doesn't fail on missing required args
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print_help()
        sys.exit(EXIT_OK)

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--txt-file",
        type=Path,
        help="Explicit path to unified_memory.txt (use with --json-file)",
    )

    parser.add_argument(
        "--json-file",
        type=Path,
        help="Explicit path to unified_memory.json (use with --txt-file)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory containing unified_memory*.{txt,json} (single-level lookup)",
    )

    args = parser.parse_args()

    explicit_mode = args.txt_file is not None and args.json_file is not None
    dir_mode = args.output_dir is not None

    # Require exactly one mode: bool equality is true when both are set or neither is set
    if explicit_mode == dir_mode:
        print(
            "Error: provide either (--txt-file AND --json-file) OR --output-dir, not both"
        )
        print_help()
        sys.exit(EXIT_USAGE)

    if dir_mode:
        if not args.output_dir.is_dir():
            print(
                f"Error: --output-dir does not exist or is not a directory: {args.output_dir}"
            )
            sys.exit(EXIT_NOINPUT)
        txt_file, json_file = resolve_from_dir(args.output_dir)
        if txt_file is None:
            print(f"Error: no unified_memory*.txt found in {args.output_dir}")
            sys.exit(EXIT_NOINPUT)
        if json_file is None:
            print(f"Error: no unified_memory*.json found in {args.output_dir}")
            sys.exit(EXIT_NOINPUT)
    else:
        txt_file = args.txt_file
        json_file = args.json_file
        if not txt_file.exists():
            print(f"Error: Could not find {txt_file}")
            sys.exit(EXIT_NOINPUT)
        if not json_file.exists():
            print(f"Error: Could not find {json_file}")
            sys.exit(EXIT_NOINPUT)

    print("Validating unified memory outputs:")
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
        sys.exit(EXIT_OK)
    else:
        print("Some validation checks failed")
        print("Failure validating unified memory outputs")
        sys.exit(EXIT_DATAERR)
