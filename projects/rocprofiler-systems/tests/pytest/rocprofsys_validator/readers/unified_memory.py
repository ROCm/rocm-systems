# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UnifiedMemoryReader — in-process validator for unified-memory profiling output.

Ported from tests/validate-unified-memory.py. Validates the text report, the JSON
migration-stats structure, and (for fault-only output) that the Perfetto trace emits
the page-fault track but not the migration-throughput track. The check functions
print the same diagnostic strings as the original script; ``validate_output_dir``
captures them so callers see identical text.
"""
from __future__ import annotations

import contextlib
import io
import json
import mmap
import re
from pathlib import Path
from typing import Any, Optional

from rocprofsys_validator.core import CheckResult, FormatReader
from rocprofsys_validator.registry import reader

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


def validate_text_output(filepath: Path) -> bool:
    """Validate the unified-memory text report (headers + migration/fault data)."""
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
        migration_headers = ["Count", "Avg Size", "Total Size", "Migration Throughput"]
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
    """Validate the parsed unified-memory JSON structure and required fields."""
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

    required_summary_fields = ["xnack_enabled", "total_page_faults", "migration_triggers"]
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

    # Lightweight track-name smoke check (mmap + find for byte sequences) that
    # avoids requiring trace_processor in every validation environment.
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
    """Resolve unified_memory*.txt and unified_memory*.json (single-level lookup)."""
    txt_matches = sorted(output_dir.glob("unified_memory*.txt"))
    json_matches = sorted(output_dir.glob("unified_memory*.json"))

    txt_file = txt_matches[0] if txt_matches else None
    json_file = json_matches[0] if json_matches else None

    if len(txt_matches) > 1:
        print(f"Warning: multiple unified_memory*.txt files in {output_dir}; using {txt_file}")
    if len(json_matches) > 1:
        print(f"Warning: multiple unified_memory*.json files in {output_dir}; using {json_file}")

    return txt_file, json_file


def _run_dir_validation(output_dir: Path) -> bool:
    """Reproduce the standalone script's --output-dir flow (prints diagnostics)."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        print(f"Error: --output-dir does not exist or is not a directory: {output_dir}")
        return False

    txt_file, json_file = resolve_from_dir(output_dir)
    if txt_file is None:
        print(f"Error: no unified_memory*.txt found in {output_dir}")
        return False
    if json_file is None:
        print(f"Error: no unified_memory*.json found in {output_dir}")
        return False

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
        return True

    print("Some validation checks failed")
    print("Failure validating unified memory outputs")
    return False


def validate_output_dir(output_dir: Path | str) -> tuple[bool, str]:
    """Validate unified-memory outputs in a directory; return (is_valid, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = _run_dir_validation(Path(output_dir))
    return ok, buf.getvalue()


@reader("unified_memory")
class UnifiedMemoryReader(FormatReader):
    """Reader/validator for a directory of unified-memory profiling outputs."""

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)

    def validate_outputs(self) -> tuple[bool, str]:
        """Run all unified-memory checks; return (is_valid, captured diagnostics)."""
        return validate_output_dir(self._dir)

    def validate(self) -> list[CheckResult]:
        ok, text = self.validate_outputs()
        return [
            CheckResult(
                passed=ok,
                validator_name="unified_memory",
                message=text.strip(),
            )
        ]

    def close(self) -> None:
        """No-op — holds no persistent resources."""
