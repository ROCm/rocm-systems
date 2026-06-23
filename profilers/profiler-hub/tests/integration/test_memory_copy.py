# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value assertions for the memory-copy round-trip."""

from __future__ import annotations

import pytest

# Expected recovered values, keyed by the dotted path the exe prints.
EXPECTED = {
    "start_timestamp": "1200000",
    "end_timestamp": "1300000",
    "dst_address": "4096",
    "src_address": "8192",
    "size": "4096",
    "name": "hipMemcpyDtoH",
    "region_name": "integration_memcpy",
    "extdata": "{test data memory copy}",
    "event.stack_id": "2",
    "event.parent_stack_id": "0",
    "event.correlation_id": "2",
    "event.event_category": "memory_copy",
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
    "src_agent_info.agent_type": "GPU",
    "src_agent_info.type_index": "0",
    "src_agent_info.name": "integration-src-gpu",
    "dst_agent_info.agent_type": "CPU",
    "dst_agent_info.type_index": "0",
    "dst_agent_info.name": "integration-dst-cpu",
}


@pytest.fixture(scope="module")
def recovered(run_launcher) -> dict[str, str]:
    return run_launcher("memory_copy_writer_reader")


@pytest.mark.parametrize("field,expected", list(EXPECTED.items()))
def test_field_matches(recovered, field, expected):
    assert field in recovered, f"missing field '{field}' in example output"
    assert (
        recovered[field] == expected
    ), f"{field}: got {recovered[field]!r}, expected {expected!r}"


def test_no_unexpected_fields(recovered):
    assert set(recovered.keys()) == set(EXPECTED.keys())
