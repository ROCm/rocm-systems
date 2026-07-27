#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# BSD sysexits.h codes — portable across platforms (os.EX_* is POSIX-only)
EXIT_OK = 0
EXIT_USAGE = 64
EXIT_DATAERR = 65
EXIT_NOINPUT = 66

PERFETTO_FAULT_TRACK_NAME = b"Unified Memory Page Faults"
PERFETTO_MIGRATION_THROUGHPUT_TRACK_NAME = b"Unified Memory Migration Throughput"

REQUIRED_MIGRATION_DIRECTIONS = [
    "host_to_device",
    "device_to_host",
    "device_to_device",
]
REQUIRED_MIGRATION_STATS = [
    "count",
    "total_size_bytes",
    "min_size_bytes",
    "max_size_bytes",
    "avg_size_bytes",
    "total_time_ns",
    "migration_throughput_gbps",
]


def print_help() -> None:
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
        - Fault-only Perfetto trace does not emit migration-throughput tracks

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


def load_json_output(filepath: Path) -> Optional[dict[str, Any]]:
    """Load a unified-memory JSON output file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error: Could not read JSON file: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return None

    if not isinstance(data, dict):
        print("Error: JSON root must be an object")
        return None

    return data


def validate_json_data(data: dict[str, Any]) -> bool:
    """
    Validates parsed JSON output for unified memory profiling.
    Checks for proper structure, required fields, and device information.

    Returns:
        bool: True if validation passes, False otherwise
    """
    if "devices" not in data:
        print("Error: Missing 'devices' array in JSON output")
        return False

    if "summary" not in data:
        print("Error: Missing 'summary' object in JSON output")
        return False

    summary = data["summary"]
    if not isinstance(summary, dict):
        print("Error: 'summary' must be an object")
        return False

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
        if not isinstance(device, dict):
            print(f"Error: Device {i} must be an object")
            return False

        required_device_fields = ["device_id", "device_name", "migrations"]
        missing = [field for field in required_device_fields if field not in device]

        if missing:
            print(f"Error: Device {i} missing fields: {missing}")
            return False

        migrations = device["migrations"]
        if not isinstance(migrations, dict):
            print(f"Error: Device {i} 'migrations' must be an object")
            return False

        for direction in REQUIRED_MIGRATION_DIRECTIONS:
            if direction not in migrations:
                print(f"Error: Device {i} missing migration direction: {direction}")
                return False

            stats = migrations[direction]
            if not isinstance(stats, dict):
                print(f"Error: Device {i}, {direction} stats must be an object")
                return False

            missing_stats = [
                field for field in REQUIRED_MIGRATION_STATS if field not in stats
            ]

            if missing_stats:
                print(f"Error: Device {i}, {direction} missing stats: {missing_stats}")
                return False

    triggers = summary["migration_triggers"]
    if not isinstance(triggers, dict):
        print("Error: 'migration_triggers' must be an object")
        return False

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


def resolve_perfetto_trace(output_dir: Path) -> Optional[Path]:
    """Resolve a Perfetto trace in output_dir when one was generated."""
    matches = sorted(output_dir.glob("perfetto-trace*.proto"))
    if not matches:
        matches = sorted(output_dir.glob("perfetto*.proto"))

    if len(matches) > 1:
        print(f"Warning: multiple Perfetto traces in {output_dir}; using {matches[0]}")

    return matches[0] if matches else None


def is_integer(value: object) -> bool:
    """Return true for JSON integer values, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def has_observed_migrations(devices: list[Any]) -> bool:
    """Return true if any device migration bucket has a positive count."""
    for device in devices:
        # The CLI calls this only after validate_json_data() succeeds. Treat
        # malformed entries as zero here so this helper stays predicate-only.
        if not isinstance(device, dict):
            continue

        migrations = device.get("migrations")
        if not isinstance(migrations, dict):
            continue

        for direction in REQUIRED_MIGRATION_DIRECTIONS:
            stats = migrations.get(direction)
            if not isinstance(stats, dict):
                continue

            count = stats.get("count", 0)
            if is_integer(count) and count > 0:
                return True

    return False


def is_fault_only_output(data: dict[str, Any]) -> bool:
    """Return true when reports contain page faults but no observed migrations."""
    devices = data.get("devices")
    summary = data.get("summary")
    if not isinstance(devices, list) or not isinstance(summary, dict):
        return False

    total_page_faults = summary.get("total_page_faults", 0)
    return (
        is_integer(total_page_faults)
        and total_page_faults > 0
        and not has_observed_migrations(devices)
    )


def validate_perfetto_fault_only_trace(output_dir: Path, data: dict[str, Any]) -> bool:
    """Validate Perfetto track names for fault-only unified-memory output."""
    if not is_fault_only_output(data):
        print("Perfetto fault-only validation skipped (migration data present)")
        return True

    perfetto_trace = resolve_perfetto_trace(output_dir)
    if perfetto_trace is None:
        print("Warning: no Perfetto trace found; skipping Perfetto track validation")
        return True

    if perfetto_trace.stat().st_size == 0:
        print("Error: Perfetto trace is empty")
        return False

    print(f"Validating Perfetto fault-only tracks: {perfetto_trace}")

    # Lightweight track-name smoke check. This avoids requiring trace_processor
    # in every validation environment while still catching accidental track
    # creation in the common case. mmap keeps large traces off the Python heap;
    # use find() because mmap's `in` operator matches single bytes, not byte
    # sequences.
    with open(perfetto_trace, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as content:
        if content.find(PERFETTO_FAULT_TRACK_NAME) == -1:
            print(
                "Error: fault-only Perfetto trace is missing unified-memory fault track"
            )
            return False

        if content.find(PERFETTO_MIGRATION_THROUGHPUT_TRACK_NAME) != -1:
            print(
                "Error: fault-only Perfetto trace contains unified-memory migration "
                "throughput track"
            )
            return False

    print("Perfetto fault-only track validation passed")
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
    print(f"Validating JSON output: {json_file}")
    json_data = load_json_output(json_file)
    json_valid = json_data is not None and validate_json_data(json_data)
    perfetto_valid = True
    if json_valid:
        assert json_data is not None
        perfetto_valid = validate_perfetto_fault_only_trace(json_file.parent, json_data)

    print()
    if txt_valid and json_valid and perfetto_valid:
        print("All validation checks passed")
        print(f"{txt_file} and {json_file} validated")
        sys.exit(EXIT_OK)
    else:
        print("Some validation checks failed")
        print("Failure validating unified memory outputs")
        sys.exit(EXIT_DATAERR)
