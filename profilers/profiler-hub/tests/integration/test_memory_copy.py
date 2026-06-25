# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate memory-copy writer data persisted to SQLite."""

from __future__ import annotations

import pytest

from profiler_hub_db import ProfilerHubDb

pytestmark = pytest.mark.timeout(120)

EXPECTED = {
    "start_timestamp": 1200000,
    "end_timestamp": 1300000,
    "dst_address": 4096,
    "src_address": 8192,
    "size": 4096,
    "name": "hipMemcpyDtoH",
    "region_name": "integration_memcpy",
    "extdata": "{test data memory copy}",
    "thread_info.thread_id": 100,
    "thread_info.name": "integration-thread",
    "thread_info.parent_process_id": 1000,
    "thread_info.start": 1000000,
    "thread_info.end": 2000000,
    "thread_info.node_info.node_id": 1,
    "thread_info.process_info.pid": 1000,
    "src_agent_info.agent_type": "GPU",
    "src_agent_info.absolute_index": 0,
    "src_agent_info.logical_index": 0,
    "src_agent_info.type_index": 0,
    "src_agent_info.uuid": 100,
    "src_agent_info.name": "integration-src-gpu",
    "src_agent_info.model_name": "integration-model",
    "src_agent_info.vendor_name": "AMD",
    "src_agent_info.product_name": "integration-product",
    "src_agent_info.user_name": "integration-src-gpu",
    "src_agent_info.node_info.node_id": 1,
    "src_agent_info.process_info.pid": 1000,
    "dst_agent_info.agent_type": "CPU",
    "dst_agent_info.absolute_index": 0,
    "dst_agent_info.logical_index": 0,
    "dst_agent_info.type_index": 0,
    "dst_agent_info.uuid": 100,
    "dst_agent_info.name": "integration-dst-cpu",
    "dst_agent_info.model_name": "integration-model",
    "dst_agent_info.vendor_name": "AMD",
    "dst_agent_info.product_name": "integration-product",
    "dst_agent_info.user_name": "integration-dst-cpu",
    "dst_agent_info.node_info.node_id": 1,
    "dst_agent_info.process_info.pid": 1000,
    "queue_info.db_id": 0,
    "queue_info.name": "integration-queue",
    "queue_info.node_info.node_id": 1,
    "queue_info.process_info.pid": 1000,
    "stream_info.db_id": 0,
    "stream_info.name": "integration-stream",
    "stream_info.node_info.node_id": 1,
    "stream_info.process_info.pid": 1000,
    "event.stack_id": 2,
    "event.parent_stack_id": 0,
    "event.correlation_id": 2,
    "event.event_category": "memory_copy",
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
    db_path = tmp_path_factory.mktemp("profiler_hub") / "memory_copy.db"
    db_path = run_launcher_db("memory_copy_writer", db_path)
    database = ProfilerHubDb.open(db_path)
    try:
        info = database.read_memory_copy_info()
    finally:
        database.close()
    yield info
    database.delete()


def test_memory_copy_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
