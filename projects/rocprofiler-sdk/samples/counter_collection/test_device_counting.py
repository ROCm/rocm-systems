#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import itertools
import math
import os
import re
import sys

import pytest

SYNC_SAMPLE_PATTERN = re.compile(r"^Sample (?P<sample>\d+):$")
SYNC_RECORD_PATTERN = re.compile(
    r"^Counter: (?P<id>\d+) Name: (?P<name>\S+) Value: (?P<value>\S+) "
    r"User data: (?P<user_data>\d+)$"
)
SYNC_DIMENSION_PATTERN = re.compile(
    r"^Dimension Name: (?P<name>.+): (?P<position>\d+)/(?P<extent>\d+)$"
)
ASYNC_RECORD_PATTERN = re.compile(
    r"\(Id: (?P<id>\d+) Value \[D\]: (?P<value>[^,]+), "
    r"user_data: (?P<user_data>\d+)\),"
)
UNAVAILABLE_MESSAGE = "Device counting unavailable: no hardware counters"
EXPECTED_COUNTER_NAMES = {"GRBM_COUNT", "SQ_WAVES"}


def _read_output(environment_variable):
    path = os.environ[environment_variable]
    with open(path, encoding="utf-8") as input_file:
        output = input_file.read()
    assert output, f"{path} is empty"
    assert output.endswith("\n"), f"{path} was not completely flushed"
    return output


def _numeric_value(value):
    result = float(value)
    assert math.isfinite(result), f"non-finite counter value: {value}"
    assert result >= 0.0, f"negative counter value: {value}"
    return result


def _skip_if_unavailable(output):
    if output.strip() != UNAVAILABLE_MESSAGE:
        assert (
            UNAVAILABLE_MESSAGE not in output
        ), "unavailable marker was mixed with sample output"
        return
    pytest.skip(UNAVAILABLE_MESSAGE)


def _assert_sample_capability_parity():
    sync_output = _read_output("ROCPROFILER_SAMPLE_SYNC_OUTPUT_FILE")
    async_output = _read_output("ROCPROFILER_SAMPLE_ASYNC_OUTPUT_FILE")
    assert (sync_output.strip() == UNAVAILABLE_MESSAGE) == (
        async_output.strip() == UNAVAILABLE_MESSAGE
    ), "sync and async sample capability results differ"


def _parse_sync_output(output):
    samples = {}
    current_sample = None
    current_record_id = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SYNC_SAMPLE_PATTERN.fullmatch(line)
        if match:
            current_sample = int(match.group("sample"))
            current_record_id = None
            assert current_sample not in samples
            samples[current_sample] = {}
            continue

        match = SYNC_RECORD_PATTERN.fullmatch(line)
        if match:
            assert current_sample is not None, f"record before sample: {line}"
            record_id = int(match.group("id"))
            current_record_id = record_id
            assert record_id not in samples[current_sample]
            samples[current_sample][record_id] = {
                "name": match.group("name"),
                "value": _numeric_value(match.group("value")),
                "user_data": int(match.group("user_data")),
                "dimensions": {},
            }
            continue

        match = SYNC_DIMENSION_PATTERN.fullmatch(line)
        if match:
            assert current_sample is not None and current_record_id is not None
            dimensions = samples[current_sample][current_record_id]["dimensions"]
            dimension_name = match.group("name")
            assert dimension_name not in dimensions
            position = int(match.group("position"))
            extent = int(match.group("extent"))
            assert 0 <= position < extent
            dimensions[dimension_name] = (position, extent)
            continue

        raise AssertionError(f"unexpected sync output: {line}")

    return samples


def _assert_complete_dimension_coverage(records, counter_name):
    assert records, f"no {counter_name} records"
    assert all(
        record["dimensions"] for record in records
    ), f"{counter_name} records without dimensions"
    extents = {}
    coordinates = set()
    for record in records:
        coordinate = []
        for name, (position, extent) in sorted(record["dimensions"].items()):
            assert extents.setdefault(name, extent) == extent
            coordinate.append((name, position))
        assert set(record["dimensions"]) == set(extents)
        coordinates.add(tuple(coordinate))
    dimension_names = sorted(extents)
    expected_coordinates = {
        tuple(zip(dimension_names, positions))
        for positions in itertools.product(
            *(range(extents[name]) for name in dimension_names)
        )
    }
    assert (
        coordinates == expected_coordinates
    ), f"{counter_name} instances do not cover every dimension coordinate"
    assert len(records) == len(expected_coordinates)


def _validate_sync_samples(samples):
    sample_ids = sorted(samples)
    assert len(sample_ids) >= 2
    assert sample_ids == list(range(1, sample_ids[-1] + 1))

    # Context startup can transiently produce an empty sample before counter records are ready.
    # Keep those attempts visible in the log, but validate consistency across populated samples.
    populated_sample_ids = [sample_id for sample_id in sample_ids if samples[sample_id]]
    assert (
        len(populated_sample_ids) >= 2
    ), "fewer than two samples contained counter records"

    first_populated_sample = populated_sample_ids[0]
    assert all(
        samples[sample_id]
        for sample_id in sample_ids
        if sample_id >= first_populated_sample
    ), "empty sync sample after counter records became available"
    expected_record_ids = set(samples[first_populated_sample])
    first_records = samples[first_populated_sample].values()
    assert {record["name"] for record in first_records} == EXPECTED_COUNTER_NAMES
    for counter_name in sorted(EXPECTED_COUNTER_NAMES):
        _assert_complete_dimension_coverage(
            [record for record in first_records if record["name"] == counter_name],
            counter_name,
        )

    positive_counters = set()
    for sample_id in populated_sample_ids:
        records = samples[sample_id]
        assert set(records) == expected_record_ids
        assert {record["name"] for record in records.values()} == EXPECTED_COUNTER_NAMES
        for record in records.values():
            assert record["user_data"] == sample_id
            if record["value"] > 0:
                positive_counters.add(record["name"])

    assert (
        positive_counters == EXPECTED_COUNTER_NAMES
    ), "counters never reported a positive value: " + ", ".join(
        sorted(EXPECTED_COUNTER_NAMES - positive_counters)
    )


def _validate_sync_output_file():
    output = _read_output("ROCPROFILER_SAMPLE_SYNC_OUTPUT_FILE")
    _skip_if_unavailable(output)
    _validate_sync_samples(_parse_sync_output(output))


def test_sync_device_counting_output():
    _assert_sample_capability_parity()
    _validate_sync_output_file()


def _validate_async_output(output, sync_records):
    records_by_sample = {}

    for line in output.splitlines():
        assert line.startswith("[buffered_callback] "), f"unexpected async output: {line}"
        payload = line[len("[buffered_callback] ") :]
        assert not re.sub(
            r"\s+", "", ASYNC_RECORD_PATTERN.sub("", payload)
        ), "unparsed async callback content: {}".format(line)
        for match in ASYNC_RECORD_PATTERN.finditer(payload):
            sample_id = int(match.group("user_data"))
            record_id = int(match.group("id"))
            records = records_by_sample.setdefault(sample_id, {})
            assert record_id not in records
            records[record_id] = _numeric_value(match.group("value"))

    sample_ids = sorted(records_by_sample)
    assert len(sample_ids) >= 2
    assert sample_ids == list(range(sample_ids[0], sample_ids[-1] + 1))

    expected_record_ids = set(records_by_sample[sample_ids[0]])
    assert expected_record_ids
    # The async records carry no counter name, so the sync run's record ids supply
    # the mapping needed to require a positive value from every counter.
    assert expected_record_ids == set(sync_records)
    positive_counters = set()
    for records in records_by_sample.values():
        assert set(records) == expected_record_ids
        for record_id, value in records.items():
            if value > 0:
                positive_counters.add(sync_records[record_id]["name"])

    assert (
        positive_counters == EXPECTED_COUNTER_NAMES
    ), "counters never reported a positive value: " + ", ".join(
        sorted(EXPECTED_COUNTER_NAMES - positive_counters)
    )


def _validate_async_output_file():
    output = _read_output("ROCPROFILER_SAMPLE_ASYNC_OUTPUT_FILE")
    _skip_if_unavailable(output)
    sync_samples = _parse_sync_output(_read_output("ROCPROFILER_SAMPLE_SYNC_OUTPUT_FILE"))
    sync_records = next(records for _, records in sorted(sync_samples.items()) if records)
    _validate_async_output(output, sync_records)


def test_async_device_counting_output():
    _assert_sample_capability_parity()
    _validate_async_output_file()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
