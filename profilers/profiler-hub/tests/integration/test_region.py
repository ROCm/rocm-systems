# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value assertions for the region round-trip."""

from __future__ import annotations

import pytest

# Expected recovered values, keyed by the dotted path the exe prints.
EXPECTED = {
    "start_timestamp": "1400000",
    "end_timestamp": "1500000",
    "name": "integration_region",
    "extdata": "{test data region}",
    "event.stack_id": "3",
    "event.parent_stack_id": "0",
    "event.correlation_id": "3",
    "event.event_category": "region",
    "event.extdata": "{test data event}",
    "sample.track.name": "integration-region-sample-track",
    "sample.track.extdata": "{test data track}",
    "sample.track.thread_id": "100",
    "arg_count": "1",
    "arg.0.position": "0",
    "arg.0.type": "int",
    "arg.0.name": "iterations",
    "arg.0.value": "64",
    "arg.0.extdata": "{test data arg}",
}


@pytest.fixture(scope="module")
def recovered(run_launcher) -> dict[str, str]:
    return run_launcher("region_writer_reader")


@pytest.mark.parametrize("field,expected", list(EXPECTED.items()))
def test_field_matches(recovered, field, expected):
    assert field in recovered, f"missing field '{field}' in example output"
    assert (
        recovered[field] == expected
    ), f"{field}: got {recovered[field]!r}, expected {expected!r}"


def test_no_unexpected_fields(recovered):
    assert set(recovered.keys()) == set(EXPECTED.keys())
