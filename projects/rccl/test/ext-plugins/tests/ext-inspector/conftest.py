# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import pytest
import json
import glob
from types import SimpleNamespace

# Required fields for inspector JSONL output validation
INSPECTOR_HEADER_FIELDS = {"id": str, "rank": int, "n_ranks": int, "nnodes": int}
INSPECTOR_METADATA_FIELDS = {
    "inspector_output_format_version": str,
    "rec_mechanism": str,
    "dump_timestamp_us": int,
    "hostname": str,
    "pid": int,
}
INSPECTOR_COLL_PERF_FIELDS = {
    "coll": str,
    "coll_sn": int,
    "coll_msg_size_bytes": int,
    "coll_exec_time_us": int,
    "coll_timing_source": str,
    "coll_algobw_gbs": (int, float),
    "coll_busbw_gbs": (int, float),
}


def validate_inspector_log_line(line):
    """Validate a single inspector JSONL line. Returns (is_valid, record_dict_or_None, error_message)."""
    try:
        record = json.loads(line.strip())
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"

    # Check top-level sections
    for section in ("header", "metadata", "coll_perf"):
        if section not in record:
            return False, record, f"Missing top-level section: '{section}'"

    # Validate header fields
    for field, expected_type in INSPECTOR_HEADER_FIELDS.items():
        if field not in record["header"]:
            return False, record, f"Missing header field: '{field}'"
        if not isinstance(record["header"][field], expected_type):
            return False, record, f"header.{field} has wrong type: expected {expected_type.__name__}, got {type(record['header'][field]).__name__}"

    # Validate metadata fields
    for field, expected_type in INSPECTOR_METADATA_FIELDS.items():
        if field not in record["metadata"]:
            return False, record, f"Missing metadata field: '{field}'"
        if not isinstance(record["metadata"][field], expected_type):
            return False, record, f"metadata.{field} has wrong type: expected {expected_type.__name__}, got {type(record['metadata'][field]).__name__}"

    # Validate coll_perf fields
    for field, expected_type in INSPECTOR_COLL_PERF_FIELDS.items():
        if field not in record["coll_perf"]:
            return False, record, f"Missing coll_perf field: '{field}'"
        if not isinstance(record["coll_perf"][field], expected_type):
            return False, record, f"coll_perf.{field} has wrong type: expected {expected_type}, got {type(record['coll_perf'][field]).__name__}"

    # Sanity checks on values
    if record["coll_perf"]["coll_exec_time_us"] < 0:
        return False, record, "coll_perf.coll_exec_time_us is negative"
    if record["coll_perf"]["coll_msg_size_bytes"] < 0:
        return False, record, "coll_perf.coll_msg_size_bytes is negative"
    if record["coll_perf"]["coll_algobw_gbs"] < 0:
        return False, record, "coll_perf.coll_algobw_gbs is negative"
    if record["coll_perf"]["coll_busbw_gbs"] < 0:
        return False, record, "coll_perf.coll_busbw_gbs is negative"
    if record["coll_perf"]["coll_timing_source"] not in ("kernel_gpu", "kernel_cpu", "collective_cpu"):
        return False, record, f"coll_perf.coll_timing_source has unexpected value: '{record['coll_perf']['coll_timing_source']}'"

    return True, record, "OK"


def validate_inspector_log_file(filepath):
    """Validate an entire inspector JSONL log file. Returns (is_valid, num_records, list_of_errors)."""
    if not os.path.exists(filepath):
        return False, 0, ["File does not exist"]

    errors = []
    num_records = 0

    with open(filepath, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            is_valid, _, error_msg = validate_inspector_log_line(line)
            if is_valid:
                num_records += 1
            else:
                errors.append(f"Line {lineno}: {error_msg}")

    if num_records == 0:
        errors.append("No valid records found in file")

    return len(errors) == 0, num_records, errors


def count_inspector_records(filepath, coll=None):
    """Count records in an inspector log file, optionally filtered by collective type."""
    if not os.path.exists(filepath):
        return 0

    count = 0
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if coll is None or record.get("coll_perf", {}).get("coll") == coll:
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return count


def validate_inspector_verbose_record(record):
    """Validate that an inspector record contains verbose event trace fields. Returns (is_valid, error_message)."""
    coll_perf = record.get("coll_perf", {})

    # Check event_trace_sn
    if "event_trace_sn" not in coll_perf:
        return False, "Missing coll_perf.event_trace_sn"
    trace_sn = coll_perf["event_trace_sn"]
    if not isinstance(trace_sn, dict):
        return False, "coll_perf.event_trace_sn is not a dict"
    for field in ("coll_start_sn", "coll_stop_sn"):
        if field not in trace_sn:
            return False, f"Missing event_trace_sn.{field}"

    # Check event_trace_ts
    if "event_trace_ts" not in coll_perf:
        return False, "Missing coll_perf.event_trace_ts"
    trace_ts = coll_perf["event_trace_ts"]
    if not isinstance(trace_ts, dict):
        return False, "coll_perf.event_trace_ts is not a dict"
    for field in ("coll_start_ts", "coll_stop_ts"):
        if field not in trace_ts:
            return False, f"Missing event_trace_ts.{field}"

    # Check kernel_events in both trace_sn and trace_ts
    if "kernel_events" in trace_sn:
        if not isinstance(trace_sn["kernel_events"], list):
            return False, "event_trace_sn.kernel_events is not a list"
        for i, ke in enumerate(trace_sn["kernel_events"]):
            for field in ("channel_id", "kernel_start_sn", "kernel_stop_sn"):
                if field not in ke:
                    return False, f"Missing event_trace_sn.kernel_events[{i}].{field}"

    if "kernel_events" in trace_ts:
        if not isinstance(trace_ts["kernel_events"], list):
            return False, "event_trace_ts.kernel_events is not a list"
        for i, ke in enumerate(trace_ts["kernel_events"]):
            for field in ("channel_id", "kernel_start_ts", "kernel_stop_ts"):
                if field not in ke:
                    return False, f"Missing event_trace_ts.kernel_events[{i}].{field}"

    return True, "OK"


# Pytest Fixture - bundles all inspector helpers into a single fixture
@pytest.fixture(scope="session")
def inspector_helpers():
    """Provides inspector-specific helper functions and constants."""
    return SimpleNamespace(
        validate_inspector_log_line=validate_inspector_log_line,
        validate_inspector_log_file=validate_inspector_log_file,
        count_inspector_records=count_inspector_records,
        validate_inspector_verbose_record=validate_inspector_verbose_record,
    )


@pytest.fixture(scope="session", autouse=True)
def clear_inspector_dump(request, paths):
    """Automatically clear inspector dump folder once before ext_inspector tests."""
    has_inspector_tests = any(
        item.get_closest_marker("ext_inspector")
        for item in request.session.items
    )

    if has_inspector_tests:
        # Clear all .log files in the inspector dump directory (including subdirectories)
        pattern = os.path.join(paths.INSPECTOR_DUMP_DIR, "**", "*.log")
        for log_file in glob.glob(pattern, recursive=True):
            try:
                os.remove(log_file)
            except OSError:
                pass  # Ignore errors if file doesn't exist or can't be removed

