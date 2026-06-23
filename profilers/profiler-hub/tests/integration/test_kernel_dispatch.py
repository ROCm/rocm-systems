# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value assertions for the kernel-dispatch round-trip.

The example binary examples/schema_v3/kernel_dispatch_writer_reader.cpp writes
one kernel dispatch, reads it back, and prints the recovered fields as
"key=value" lines. Python owns the expected values below and asserts every
reader-recoverable field against them.
"""

from __future__ import annotations

import pytest

# Expected recovered values, keyed by the dotted path the exe prints.
EXPECTED = {
    "dispatch_id": "7",
    "start_timestamp": "1000000",
    "end_timestamp": "2000000",
    "private_segment_size": "0",
    "group_segment_size": "256",
    "workgroup_size_x": "128",
    "workgroup_size_y": "1",
    "workgroup_size_z": "1",
    "grid_size_x": "4096",
    "grid_size_y": "1",
    "grid_size_z": "1",
    "name": "integration_kernel",
    "extdata": "{test data kernel}",
    "event.stack_id": "1",
    "event.parent_stack_id": "0",
    "event.correlation_id": "1",
    "event.event_category": "kernel_dispatch",
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
    "thread_info.thread_id": "100",
    "thread_info.name": "integration-thread",
    "code_object_info.id": "1",
    "code_object_info.uri": "file:///integration/kernel.co",
    "kernel_symbol_info.id": "1",
    "kernel_symbol_info.name": "integration_kernel",
    "kernel_symbol_info.display_name": "Integration Kernel",
}


@pytest.fixture(scope="module")
def recovered(run_launcher) -> dict[str, str]:
    return run_launcher("kernel_dispatch_writer_reader")


@pytest.mark.parametrize("field,expected", list(EXPECTED.items()))
def test_field_matches(recovered, field, expected):
    assert field in recovered, f"missing field '{field}' in example output"
    assert (
        recovered[field] == expected
    ), f"{field}: got {recovered[field]!r}, expected {expected!r}"


def test_no_unexpected_fields(recovered):
    # Guard against the example silently dropping, adding, or renaming fields.
    assert set(recovered.keys()) == set(EXPECTED.keys())
