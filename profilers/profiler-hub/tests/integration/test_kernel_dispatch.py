# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate kernel-dispatch writer data persisted to SQLite."""

from __future__ import annotations

import pytest

from profiler_hub_db import ProfilerHubDb

pytestmark = pytest.mark.timeout(120)

# Values below are hand-transcribed from the record built by
# examples/schema_v3/kernel_dispatch_writer.cpp (see its make_node/make_process/
# make_thread/make_agent/make_queue/make_stream/make_code_object/
# make_kernel_symbol/make_kernel/make_env functions). If that writer changes,
# update this dict to match.
EXPECTED = {
    "dispatch_id": 7,
    "start_timestamp": 1000000,
    "end_timestamp": 2000000,
    "private_segment_size": 0,
    "group_segment_size": 256,
    "workgroup_size_x": 128,
    "workgroup_size_y": 1,
    "workgroup_size_z": 1,
    "grid_size_x": 4096,
    "grid_size_y": 1,
    "grid_size_z": 1,
    "name": "integration_kernel",
    "extdata": "{test data kernel}",
    "event.stack_id": 1,
    "event.parent_stack_id": 0,
    "event.correlation_id": 1,
    "event.event_category": "kernel_dispatch",
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
    "thread_info.thread_id": 100,
    "thread_info.name": "integration-thread",
    "thread_info.parent_process_id": 1000,
    "thread_info.start": 1000000,
    "thread_info.end": 2000000,
    "thread_info.node_info.node_id": 1,
    "thread_info.process_info.pid": 1000,
    "agent_info.agent_type": "GPU",
    "agent_info.absolute_index": 0,
    "agent_info.logical_index": 0,
    "agent_info.type_index": 0,
    "agent_info.uuid": 12345,
    "agent_info.name": "gfx1100",
    "agent_info.model_name": "AMD Radeon",
    "agent_info.vendor_name": "AMD",
    "agent_info.product_name": "Radeon",
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
    "code_object_info.id": 1,
    "code_object_info.uri": "file:///integration/kernel.co",
    "code_object_info.load_base": 4096,
    "code_object_info.load_size": 8192,
    "code_object_info.load_delta": 0,
    "code_object_info.storage_type": "FILE",
    "code_object_info.node_info.node_id": 1,
    "code_object_info.process_info.pid": 1000,
    "code_object_info.agent_info.agent_type": "GPU",
    "code_object_info.agent_info.type_index": 0,
    "kernel_symbol_info.id": 1,
    "kernel_symbol_info.name": "integration_kernel",
    "kernel_symbol_info.display_name": "Integration Kernel",
    "kernel_symbol_info.kernel_object": 4096,
    "kernel_symbol_info.kernarg_segment_size": 64,
    "kernel_symbol_info.kernarg_segment_alignment": 8,
    "kernel_symbol_info.group_segment_size": 256,
    "kernel_symbol_info.private_segment_size": 0,
    "kernel_symbol_info.sgpr_count": 32,
    "kernel_symbol_info.arch_vgpr_count": 64,
    "kernel_symbol_info.accum_vgpr_count": 0,
    "kernel_symbol_info.node_info.node_id": 1,
    "kernel_symbol_info.process_info.pid": 1000,
    "kernel_symbol_info.code_object_info.id": 1,
}


@pytest.fixture(scope="module")
def read_info(run_launcher_db, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("profiler_hub") / "kernel_dispatch.db"
    db_path = run_launcher_db("kernel_dispatch_writer", db_path)
    database = ProfilerHubDb.open(db_path)
    try:
        info = database.read_kernel_dispatch_info()
    finally:
        database.close()
    yield info
    database.delete()


def test_kernel_dispatch_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
