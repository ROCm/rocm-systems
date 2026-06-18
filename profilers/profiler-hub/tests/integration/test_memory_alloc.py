# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value assertions for the memory-alloc round-trip.

The example binary examples/schema_v3/memory_alloc_writer_reader.cpp writes one
memory allocation, reads it back, and prints the recovered fields as "key=value"
lines. Python owns the expected values below and asserts every reader-recoverable
field against them.
"""

from __future__ import annotations

import pytest

# Expected recovered values, keyed by the dotted path the exe prints.
EXPECTED = {
    "type": "ALLOC",
    "level": "SCRATCH",
    "start_timestamp": "1000000",
    "end_timestamp": "1100000",
    "address": "0",
    "size": "8192",
    "extdata": "{test data alloc}",
    "event.stack_id": "1",
    "event.parent_stack_id": "0",
    "event.correlation_id": "1",
    "event.event_category": "SCRATCH_MEMORY",
    "event.extdata": "{test data event}",
    "node_info.node_id": "1",
    "node_info.hash": "123456789",
    "node_info.machine_id": "integration-machine",
    "node_info.system_name": "Linux",
    "node_info.hostname": "integration-host",
    "node_info.release": "6.0.0",
    "node_info.version": "#1 SMP",
    "node_info.hardware_name": "x86_64",
    "node_info.domain_name": "integration",
    "process_info.pid": "1000",
    "process_info.ppid": "1",
    "process_info.node_info.node_id": "1",
}


@pytest.fixture(scope="module")
def recovered(run_launcher) -> dict[str, str]:
    return run_launcher("memory_alloc_writer_reader")


# One assertion per field, parametrized so each field shows up as its own
# pass/fail case in the pytest report.
@pytest.mark.parametrize("field,expected", list(EXPECTED.items()))
def test_field_matches(recovered, field, expected):
    assert field in recovered, f"missing field '{field}' in launcher output"
    assert (
        recovered[field] == expected
    ), f"{field}: got {recovered[field]!r}, expected {expected!r}"


def test_no_unexpected_fields(recovered):
    # Guard against the launcher silently dropping, adding, or renaming fields.
    assert set(recovered.keys()) == set(EXPECTED.keys())
