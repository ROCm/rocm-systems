# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate region writer data persisted to SQLite."""

from __future__ import annotations

import pytest

from profiler_hub_db import COMMON_METADATA_EXPECTED

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
    "thread_info.start": 1000000,
    "thread_info.end": 2000000,
    "event.stack_id": 3,
    "event.parent_stack_id": 0,
    "event.correlation_id": 3,
    "event.event_category": "region",
    "event.extdata": "{test data event}",
    "event.call_stack": {
        "frames": [
            {
                "address_range": {
                    "address_base": 4096,
                    "address_high": 8192,
                    "address_low": 4096,
                },
                "program_counter": {
                    "filename": "region_writer.cpp",
                    "function": "integration_region",
                    "line_number": 42,
                },
            }
        ]
    },
    "event.line_info": {
        "entries": [
            {
                "program_counter": {
                    "filename": "region_writer.cpp",
                    "function": "integration_region",
                    "line_number": 40,
                },
                "source_code": {
                    "filename": "region_writer.cpp",
                    "source_code_lines": ["insert_region"],
                    "starting_line_number": 40,
                },
            }
        ]
    },
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
    **COMMON_METADATA_EXPECTED,
}


@pytest.fixture(scope="module")
def read_info(read_example_db):
    return read_example_db("region_writer", "read_region_info", "region.db")


def test_region_db_matches_writer_fixture(read_info):
    assert read_info == EXPECTED
