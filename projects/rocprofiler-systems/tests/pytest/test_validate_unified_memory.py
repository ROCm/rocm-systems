# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the standalone unified-memory validator."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rocprofsys.validators import ValidationResult, validate_unified_memory_outputs

pytestmark = [pytest.mark.validation_usm]


@pytest.fixture
def migration_stats() -> dict[str, int]:
    """Return a zero-count migration stats object."""
    return {
        "count": 0,
        "total_size_bytes": 0,
        "min_size_bytes": 0,
        "max_size_bytes": 0,
        "avg_size_bytes": 0,
        "total_time_ns": 0,
        "migration_throughput_gbps": 0,
    }


@pytest.fixture
def valid_data(migration_stats: dict[str, int]) -> dict[str, Any]:
    """Return a valid fault-only unified-memory JSON object."""
    return {
        "summary": {
            "xnack_enabled": True,
            "total_page_faults": 1,
            "migration_triggers": {
                "gpu_page_fault": 1,
                "cpu_page_fault": 0,
                "prefetch": 0,
                "ttm_eviction": 0,
                "unknown": 0,
            },
        },
        "devices": [
            {
                "device_id": 0,
                "device_name": "gfx-test",
                "migrations": {
                    "host_to_device": dict(migration_stats),
                    "device_to_host": dict(migration_stats),
                    "device_to_device": dict(migration_stats),
                },
            }
        ],
    }


def write_text_output(output_dir: Path) -> None:
    """Write a valid fault-only unified-memory text report."""
    (output_dir / "unified_memory.txt").write_text(
        "Unified Memory profiling result\nTotal Page Faults: 1\n",
        encoding="utf-8",
    )


def write_json_output(output_dir: Path, payload: dict[str, Any]) -> None:
    """Write a unified-memory JSON report."""
    (output_dir / "unified_memory.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def run_validator(tests_dir: Path, output_dir: Path) -> ValidationResult:
    """Run the installed-style unified-memory validator wrapper."""
    return validate_unified_memory_outputs(output_dir, tests_dir=tests_dir)


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("summary", [], "'summary' must be an object"),
        ("devices", {}, "'devices' must be an array"),
    ],
)
def test_validate_json_data_rejects_top_level_type_mismatches(
    tests_dir: Path,
    tmp_path: Path,
    valid_data: dict[str, Any],
    field: str,
    value: object,
    expected_message: str,
) -> None:
    """Nested validation reports wrong top-level types without throwing."""
    payload = {**valid_data, field: value}
    write_text_output(tmp_path)
    write_json_output(tmp_path, payload)

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is False
    assert expected_message in result.message


@pytest.mark.parametrize(
    "mutator,expected_message",
    [
        pytest.param(
            lambda data: data["summary"].__setitem__("migration_triggers", []),
            "'migration_triggers' must be an object",
            id="migration-triggers-not-object",
        ),
        pytest.param(
            lambda data: data["devices"].__setitem__(0, "not-an-object"),
            "Device 0 must be an object",
            id="device-entry-not-object",
        ),
        pytest.param(
            lambda data: data["devices"][0].__setitem__("migrations", []),
            "Device 0 'migrations' must be an object",
            id="migrations-not-object",
        ),
        pytest.param(
            lambda data: data["devices"][0]["migrations"].__setitem__(
                "host_to_device",
                [],
            ),
            "Device 0, host_to_device stats must be an object",
            id="direction-stats-not-object",
        ),
    ],
)
def test_validate_json_data_rejects_nested_type_mismatches(
    tests_dir: Path,
    tmp_path: Path,
    valid_data: dict[str, Any],
    mutator: Callable[[dict[str, Any]], None],
    expected_message: str,
) -> None:
    """Nested validation reports malformed object fields without throwing."""
    payload = json.loads(json.dumps(valid_data))
    mutator(payload)
    write_text_output(tmp_path)
    write_json_output(tmp_path, payload)

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is False
    assert expected_message in result.message


def test_validate_json_data_accepts_valid_fault_only_data(
    tests_dir: Path,
    tmp_path: Path,
    valid_data: dict[str, Any],
) -> None:
    """A valid zero-migration fault-only JSON object remains accepted."""
    write_text_output(tmp_path)
    write_json_output(tmp_path, valid_data)

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is True
    assert "All validation checks passed" in result.message


@pytest.mark.parametrize(
    "filename,writer,expected_message",
    [
        pytest.param(
            None,
            lambda path: path.mkdir(),
            "Could not read JSON file",
            id="directory-path",
        ),
        pytest.param(
            "unified_memory.json",
            lambda path: path.write_bytes(b"\xff"),
            "Could not read JSON file",
            id="invalid-encoding",
        ),
        pytest.param(
            "unified_memory.json",
            lambda path: path.write_text("{", encoding="utf-8"),
            "Invalid JSON format",
            id="invalid-json",
        ),
    ],
)
def test_load_json_output_reports_read_failures(
    tests_dir: Path,
    tmp_path: Path,
    filename: str | None,
    writer: Callable[[Path], object],
    expected_message: str,
) -> None:
    """JSON loader reports read, encoding, and parse failures cleanly."""
    write_text_output(tmp_path)
    json_path = (
        tmp_path / "unified_memory.json" if filename is None else tmp_path / filename
    )
    writer(json_path)

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is False
    assert expected_message in result.message


@pytest.mark.parametrize(
    "count,expected_valid,expected_message",
    [
        (0, False, "contains unified-memory migration throughput track"),
        (1, True, "Perfetto fault-only validation skipped"),
        (True, False, "contains unified-memory migration throughput track"),
    ],
)
def test_is_fault_only_output_depends_on_positive_integer_migrations(
    tests_dir: Path,
    tmp_path: Path,
    valid_data: dict[str, Any],
    count: object,
    expected_valid: bool,
    expected_message: str,
) -> None:
    """Only positive JSON integer migration counts make output non-fault-only."""
    payload = json.loads(json.dumps(valid_data))
    payload["devices"][0]["migrations"]["host_to_device"]["count"] = count
    write_text_output(tmp_path)
    write_json_output(tmp_path, payload)
    (tmp_path / "perfetto-trace.proto").write_bytes(
        b"Unified Memory Page Faults Unified Memory Migration Throughput"
    )

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is expected_valid
    assert expected_message in result.message


@pytest.mark.parametrize(
    "trace_content,expected_valid,expected_message",
    [
        (
            b"prefix Unified Memory Page Faults suffix",
            True,
            "All validation checks passed",
        ),
        (
            b"Unified Memory Page Faults Unified Memory Migration Throughput",
            False,
            "contains unified-memory migration throughput track",
        ),
        (b"", False, "Perfetto trace is empty"),
    ],
)
def test_validate_perfetto_fault_only_trace_checks_track_names(
    tests_dir: Path,
    tmp_path: Path,
    valid_data: dict[str, Any],
    trace_content: bytes,
    expected_valid: bool,
    expected_message: str,
) -> None:
    """Fault-only traces require the fault track and reject throughput tracks."""
    write_text_output(tmp_path)
    write_json_output(tmp_path, valid_data)
    (tmp_path / "perfetto-trace.proto").write_bytes(trace_content)

    result = run_validator(tests_dir, tmp_path)

    assert result.is_valid is expected_valid
    assert expected_message in result.message
