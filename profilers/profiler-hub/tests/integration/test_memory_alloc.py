# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate memory-alloc writer data persisted to SQLite.

The example binary writes one deterministic memory allocation. Python opens the
generated DB and validates the rows and metadata relationships written by the
writer API.
"""

from __future__ import annotations

import pytest

from profiler_hub_db import ProfilerHubDb

EXPECTED = {
    "type": "ALLOC",
    "level": "SCRATCH",
    "start_timestamp": 1000000,
    "end_timestamp": 1100000,
    "address": 0,
    "size": 8192,
    "extdata": "{test data alloc}",
    "thread_info.thread_id": 100,
    "thread_info.name": "integration-thread",
    "thread_info.parent_process_id": 1000,
    "thread_info.start": 900000,
    "thread_info.end": 1200000,
    "thread_info.node_info.node_id": 1,
    "thread_info.process_info.pid": 1000,
    "agent_info.agent_type": "GPU",
    "agent_info.absolute_index": 0,
    "agent_info.logical_index": 0,
    "agent_info.type_index": 0,
    "agent_info.uuid": 42,
    "agent_info.name": "integration-gpu",
    "agent_info.model_name": "gfx-integration",
    "agent_info.vendor_name": "AMD",
    "agent_info.product_name": "Integration GPU",
    "agent_info.user_name": "gpu0",
    "agent_info.node_info.node_id": 1,
    "agent_info.process_info.pid": 1000,
    "queue_info.db_id": 0,
    "queue_info.name": "integration-queue",
    "queue_info.node_info.node_id": 1,
    "queue_info.process_info.pid": 1000,
    "stream_info.db_id": 0,
    "stream_info.name": "integration-stream",
    "stream_info.node_info.node_id": 1,
    "stream_info.process_info.pid": 1000,
    "event.stack_id": 1,
    "event.parent_stack_id": 0,
    "event.correlation_id": 1,
    "event.event_category": "SCRATCH_MEMORY",
    "event.extdata": "{test data event}",
    "node_info.node_id": 1,
    "node_info.hash": 123456789,
    "node_info.machine_id": "integration-machine",
    "node_info.system_name": "Linux",
    "node_info.hostname": "integration-host",
    "node_info.release": "6.0.0",
    "node_info.version": "#1 SMP",
    "node_info.hardware_name": "x86_64",
    "node_info.domain_name": "integration",
    "process_info.pid": 1000,
    "process_info.ppid": 1,
    "process_info.node_info.node_id": 1,
}


@pytest.fixture(scope="module")
def read_info(run_launcher_db, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("profiler_hub") / "memory_alloc.db"
    db_path = run_launcher_db("memory_alloc_writer", db_path)
    database = ProfilerHubDb.open(db_path)
    try:
        info = database.read_memory_alloc_info()
    finally:
        database.close()
    yield info
    database.delete()


def test_memory_alloc_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
