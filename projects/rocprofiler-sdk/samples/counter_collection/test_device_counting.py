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

import math
import os
import re

SYNC_SAMPLE_PATTERN = re.compile(r"^Sample (?P<sample>\d+):$")
SYNC_RECORD_PATTERN = re.compile(
    r"^Counter: (?P<id>\d+) Name: (?P<name>\S+) Value: (?P<value>\S+) "
    r"User data: (?P<user_data>\d+)$"
)
SYNC_DIMENSION_PATTERN = re.compile(r"^Dimension Name: (?P<name>.+): (?P<position>\d+)$")
ASYNC_RECORD_PATTERN = re.compile(
    r"\(Id: (?P<id>\d+) Value \[D\]: (?P<value>[^,]+), "
    r"user_data: (?P<user_data>\d+)\),"
)
UNAVAILABLE_MESSAGE = "Device counting unavailable: no hardware counters"


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
        return False
    try:
        import pytest
    except ImportError:
        print("SKIP: {}".format(UNAVAILABLE_MESSAGE))
        return True
    pytest.skip(UNAVAILABLE_MESSAGE)
    return True


def _parse_sync_output(output):
    samples = {}
    current_sample = None
    current_record = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SYNC_SAMPLE_PATTERN.fullmatch(line)
        if match:
            current_sample = int(match.group("sample"))
            assert current_sample not in samples
            samples[current_sample] = {}
            current_record = None
            continue

        match = SYNC_RECORD_PATTERN.fullmatch(line)
        if match:
            assert current_sample is not None, f"record before sample: {line}"
            record_id = int(match.group("id"))
            assert record_id not in samples[current_sample]
            samples[current_sample][record_id] = {
                "name": match.group("name"),
                "value": _numeric_value(match.group("value")),
                "user_data": int(match.group("user_data")),
                "dimensions": {},
            }
            current_record = samples[current_sample][record_id]
            continue

        match = SYNC_DIMENSION_PATTERN.fullmatch(line)
        if match:
            assert current_record is not None, f"dimension before record: {line}"
            dimension_name = match.group("name")
            assert dimension_name not in current_record["dimensions"]
            current_record["dimensions"][dimension_name] = int(match.group("position"))
            continue

        raise AssertionError(f"unexpected sync output: {line}")

    return samples


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
    first_populated_index = sample_ids.index(first_populated_sample)
    assert all(not samples[sample_id] for sample_id in sample_ids[:first_populated_index])
    assert all(samples[sample_id] for sample_id in sample_ids[first_populated_index:])

    expected_record_ids = set(samples[first_populated_sample])
    found_positive_value = False
    for sample_id in populated_sample_ids:
        records = samples[sample_id]
        assert set(records) == expected_record_ids
        for record in records.values():
            assert record["name"] == "SQ_WAVES"
            assert record["user_data"] == sample_id
            found_positive_value = found_positive_value or record["value"] > 0

    for record in samples[first_populated_sample].values():
        assert record["dimensions"], f"record has no dimensions: {record}"
        assert all(name for name in record["dimensions"])
        assert all(position >= 0 for position in record["dimensions"].values())
    assert found_positive_value, "no populated sample contained a positive SQ_WAVES value"


def test_sync_device_counting_output():
    output = _read_output("ROCPROFILER_SAMPLE_SYNC_OUTPUT_FILE")
    if _skip_if_unavailable(output):
        return
    samples = _parse_sync_output(output)
    _validate_sync_samples(samples)


def test_sync_device_counting_output_allows_leading_empty_attempts():
    output = """\
Sample 1:
Sample 2:
Sample 3:
Counter: 1 Name: SQ_WAVES Value: 3 User data: 3
Dimension Name: DIMENSION_XCC: 0
Sample 4:
Counter: 1 Name: SQ_WAVES Value: 4 User data: 4
"""
    _validate_sync_samples(_parse_sync_output(output))


def test_unavailable_device_counting_is_skipped():
    import pytest

    with pytest.raises(pytest.skip.Exception):
        _skip_if_unavailable(UNAVAILABLE_MESSAGE)


def _validate_async_output(output):
    records_by_sample = {}
    saw_populated_callback = False

    for line in output.splitlines():
        assert line.startswith("[buffered_callback] "), f"unexpected async output: {line}"
        matches = list(ASYNC_RECORD_PATTERN.finditer(line))
        # An asynchronous buffer callback can contain no value records while the service starts.
        # Require multiple populated samples below instead of treating an empty callback as fatal.
        if not matches:
            assert (
                not saw_populated_callback
            ), "empty async callback followed populated records"
            continue
        saw_populated_callback = True
        for match in matches:
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
    found_positive_value = False
    for records in records_by_sample.values():
        assert set(records) == expected_record_ids
        found_positive_value = found_positive_value or any(
            value > 0 for value in records.values()
        )
    assert found_positive_value, "no async sample contained a positive SQ_WAVES value"


def test_async_device_counting_output():
    output = _read_output("ROCPROFILER_SAMPLE_ASYNC_OUTPUT_FILE")
    if _skip_if_unavailable(output):
        return
    _validate_async_output(output)


def test_async_device_counting_output_allows_leading_empty_callbacks():
    output = (
        "[buffered_callback] \n"
        "[buffered_callback] \n"
        "[buffered_callback] (Id: 1 Value [D]: 3, user_data: 2),\n"
        "[buffered_callback] (Id: 1 Value [D]: 4, user_data: 3),\n"
    )
    _validate_async_output(output)


if __name__ == "__main__":
    test_sync_device_counting_output()
    test_async_device_counting_output()
