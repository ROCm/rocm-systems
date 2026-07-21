# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate PMC-event writer data persisted to SQLite."""

from __future__ import annotations

import pytest

from profiler_hub_db import ProfilerHubDb

pytestmark = pytest.mark.timeout(120)

# Values below are hand-transcribed from the record built by
# examples/schema_v3/pmc_event_writer.cpp (see its make_node/make_process/
# make_thread/make_agent/make_track/make_pmc_info/make_event/make_sample/
# make_pmc_event functions). If that writer changes, update this dict to
# match.
EXPECTED = {
    "value": 1234.5,
    "extdata": "{test data pmc event}",
    "pmc_info.target_arch": "GPU",
    "pmc_info.event_code": 42,
    "pmc_info.instance_id": 7,
    "pmc_info.name": "SQ_WAVES",
    "pmc_info.symbol": "SQ_WAVES",
    "pmc_info.description": "Wavefronts launched",
    "pmc_info.long_description": "Number of wavefronts launched by shader queues",
    "pmc_info.component": "SQ",
    "pmc_info.units": "waves",
    "pmc_info.value_type": "ABS",
    "pmc_info.block": "SQ",
    "pmc_info.expression": "SQ_WAVES",
    "pmc_info.is_constant": 0,
    "pmc_info.is_derived": 0,
    "pmc_info.extdata": "{test data pmc info}",
    "pmc_info.node_info.node_id": 1,
    "pmc_info.process_info.pid": 1000,
    "pmc_info.agent_info.agent_type": "GPU",
    "pmc_info.agent_info.type_index": 0,
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
    "event.stack_id": 5,
    "event.parent_stack_id": 0,
    "event.correlation_id": 5,
    "event.event_category": "pmc_event",
    "event.extdata": "{test data event}",
    "sample.timestamp": 1600000,
    "sample.extdata": "{test data sample}",
    "sample.track.name": "integration-pmc-sample-track",
    "sample.track.extdata": "{test data track}",
    "sample.track.node_info.node_id": 1,
    "sample.track.process_info.pid": 1000,
    "sample.track.thread_id": 100,
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
    "thread_info.parent_process_id": 1000,
    "thread_info.start": 1000000,
    "thread_info.end": 2000000,
    "thread_info.node_info.node_id": 1,
    "thread_info.process_info.pid": 1000,
}


@pytest.fixture(scope="module")
def read_info(run_launcher_db, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("profiler_hub") / "pmc_event.db"
    db_path = run_launcher_db("pmc_event_writer", db_path)
    database = ProfilerHubDb.open(db_path)
    try:
        info = database.read_pmc_event_info()
    finally:
        database.close()
    yield info
    database.delete()


def test_pmc_event_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
