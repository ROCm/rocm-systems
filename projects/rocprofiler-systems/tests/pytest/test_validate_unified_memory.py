# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the standalone unified-memory validator."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.validation_usm]


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    """Import validate-unified-memory.py as a test module."""
    script = Path(__file__).resolve().parents[1] / "validate-unified-memory.py"
    spec = spec_from_file_location("validate_unified_memory", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator module from {script}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("summary", []),
        ("devices", {}),
    ],
)
def test_validate_json_data_rejects_top_level_type_mismatches(
    validator: ModuleType,
    valid_data: dict[str, Any],
    field: str,
    value: object,
) -> None:
    """Nested validation reports wrong top-level types without throwing."""
    payload = {**valid_data, field: value}

    assert validator.validate_json_data(payload) is False


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(
            lambda data: data["summary"].__setitem__("migration_triggers", []),
            id="migration-triggers-not-object",
        ),
        pytest.param(
            lambda data: data["devices"].__setitem__(0, "not-an-object"),
            id="device-entry-not-object",
        ),
        pytest.param(
            lambda data: data["devices"][0].__setitem__("migrations", []),
            id="migrations-not-object",
        ),
        pytest.param(
            lambda data: data["devices"][0]["migrations"].__setitem__(
                "host_to_device",
                [],
            ),
            id="direction-stats-not-object",
        ),
    ],
)
def test_validate_json_data_rejects_nested_type_mismatches(
    validator: ModuleType,
    valid_data: dict[str, Any],
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    """Nested validation reports malformed object fields without throwing."""
    payload = json.loads(json.dumps(valid_data))
    mutator(payload)

    assert validator.validate_json_data(payload) is False


def test_validate_json_data_accepts_valid_fault_only_data(
    validator: ModuleType,
    valid_data: dict[str, Any],
) -> None:
    """A valid zero-migration fault-only JSON object remains accepted."""
    assert validator.validate_json_data(valid_data) is True


@pytest.mark.parametrize(
    "filename,writer",
    [
        pytest.param(
            None,
            lambda path: path,
            id="directory-path",
        ),
        pytest.param(
            "invalid-encoding.json",
            lambda path: path.write_bytes(b"\xff"),
            id="invalid-encoding",
        ),
        pytest.param(
            "invalid-json.json",
            lambda path: path.write_text("{", encoding="utf-8"),
            id="invalid-json",
        ),
    ],
)
def test_load_json_output_reports_read_failures(
    validator: ModuleType,
    tmp_path: Path,
    filename: str | None,
    writer: Callable[[Path], object],
) -> None:
    """JSON loader returns None for read, encoding, and parse failures."""
    json_path = tmp_path if filename is None else tmp_path / filename
    writer(json_path)

    assert validator.load_json_output(json_path) is None


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, True),
        (1, False),
        (True, True),
    ],
)
def test_is_fault_only_output_depends_on_positive_integer_migrations(
    validator: ModuleType,
    valid_data: dict[str, Any],
    count: object,
    expected: bool,
) -> None:
    """Only positive JSON integer migration counts make output non-fault-only."""
    payload = json.loads(json.dumps(valid_data))
    payload["devices"][0]["migrations"]["host_to_device"]["count"] = count

    assert validator.is_fault_only_output(payload) is expected


@pytest.mark.parametrize(
    "trace_content,expected",
    [
        (b"prefix Unified Memory Page Faults suffix", True),
        (
            b"Unified Memory Page Faults Unified Memory Migration Throughput",
            False,
        ),
        (b"", False),
    ],
)
def test_validate_perfetto_fault_only_trace_checks_track_names(
    validator: ModuleType,
    tmp_path: Path,
    valid_data: dict[str, Any],
    trace_content: bytes,
    expected: bool,
) -> None:
    """Fault-only traces require the fault track and reject throughput tracks."""
    (tmp_path / "perfetto-trace.proto").write_bytes(trace_content)

    assert validator.validate_perfetto_fault_only_trace(tmp_path, valid_data) is expected
