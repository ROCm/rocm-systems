# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate region writer data persisted to SQLite."""

from __future__ import annotations

import pytest

from profiler_hub_db import ProfilerHubDb

pytestmark = pytest.mark.timeout(120)

# Values below are hand-transcribed from the record built by
# examples/schema_v3/region_writer.cpp (see its make_node/make_process/
# make_thread/make_track/make_event/make_arg/make_region/make_env functions).
# If that writer changes, update this dict to match.
EXPECTED = {
    "start_timestamp": 1400000,
    "end_timestamp": 1500000,
    "name": "integration_region",
    "extdata": "{test data region}",
    "thread_info.thread_id": 100,
    "thread_info.name": "integration-thread",
    "thread_info.parent_process_id": 1000,
    "thread_info.start": 1000000,
    "thread_info.end": 2000000,
    "thread_info.node_info.node_id": 1,
    "thread_info.process_info.pid": 1000,
    "event.stack_id": 3,
    "event.parent_stack_id": 0,
    "event.correlation_id": 3,
    "event.event_category": "region",
    "event.extdata": "{test data event}",
    "arg_count": 1,
    "arg.0.position": 0,
    "arg.0.type": "int",
    "arg.0.name": "iterations",
    "arg.0.value": "64",
    "arg.0.extdata": "{test data arg}",
    "sample.timestamp": 1400000,
    "sample.extdata": "{}",
    "sample.track.name": "integration-region-sample-track",
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
}


@pytest.fixture(scope="module")
def read_info(run_launcher_db, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("profiler_hub") / "region.db"
    db_path = run_launcher_db("region_writer", db_path)
    database = ProfilerHubDb.open(db_path)
    try:
        info = database.read_region_info()
    finally:
        database.close()
    yield info
    database.delete()


def test_region_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
