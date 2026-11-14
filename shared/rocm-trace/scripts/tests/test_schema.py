# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""The probe schema (<output_name>.json): it must describe exactly the events
and fields the generated probes write, so a reader can rebuild the arguments."""

import json
import re

import pytest

from conftest import DATA, generate_files

# The fields a probe writes, by event: TraceLoggingWrite(provider, "<event>", then one
# TraceLogging<Type>(..., "<field>") per line; level, keyword and opcode are not fields.
_WRITE = re.compile(r'TraceLoggingWrite\(\w+, "(\w+)",\n(.*?\);)$', re.M | re.S)
_FIELD = re.compile(r'"(\w+)"\)[,;]?\)?;?$')
_NOT_FIELDS = ("TraceLoggingLevel(", "TraceLoggingKeyword(", "TraceLoggingOpcode(")


def _written_fields(source):
    events = {}
    for name, body in _WRITE.findall(source):
        fields = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith(_NOT_FIELDS):
                continue
            match = _FIELD.search(line)
            assert match, line
            fields.append(match.group(1))
        assert name not in events, f"{name} written twice"
        events[name] = fields
    return events


def assert_schema_matches_source(files, stem):
    schema = json.loads(files[f"{stem}.json"])
    written = _written_fields(files[f"{stem}.cpp"])
    described = {e["name"]: [f["name"] for f in e["fields"]] for e in schema["events"]}
    assert described == written


@pytest.fixture(scope="module")
def files():
    return generate_files(DATA / "fake.yaml")


@pytest.fixture(scope="module")
def schema(files):
    return json.loads(files["fake_probes.json"])


def _field(schema, event, name):
    for e in schema["events"]:
        if e["name"] == event:
            for f in e["fields"]:
                if f["name"] == name:
                    return f
    raise AssertionError(f"{event}.{name} not in the schema")


def test_schema_describes_every_written_field(files):
    assert_schema_matches_source(files, "fake_probes")


def test_schema_header(schema):
    assert schema["schema"] == "rocm-trace-probe-schema" and schema["version"] == 1
    assert schema["provider"] == "fake_provider" and schema["config"] == "fake.yaml"
    assert schema["capture_levels"] == ["none", "scalars", "full"]
    assert {"api": "fakeSkipped", "table": "FakeTable"}.items() <= schema["skipped"][
        0
    ].items()


def test_events_carry_level_keywords_and_opcode(schema):
    alloc = {e["name"]: e for e in schema["events"]}
    assert alloc["fakeAlloc_enter"]["keywords"] == ["API", "ALLOC"]
    assert alloc["fakeAlloc_enter"]["opcode"] == "start"
    assert alloc["fakeAlloc_exit"]["opcode"] == "stop"
    assert alloc["fakeLoad_enter"]["level"] == "verbose"


def test_enums_name_their_enumerators(schema):
    result = _field(schema, "fakeName_exit", "result")
    assert result["enum"] == "fakeError_t" and result["role"] == "result"
    enumerators = schema["enums"]["fakeError_t"]["enumerators"]
    assert enumerators == {"fakeSuccess": 0, "fakeErrorInvalidValue": 1}
    assert schema["enums"]["fakeWide"]["size"] == 8


def test_flattened_structs_keep_their_member_path(schema):
    grid_x = _field(schema, "fakeLaunchEx_enter", "config_grid_x")
    assert grid_x["param"] == "config" and grid_x["path"] == ["grid", "x"]
    assert grid_x["param_type"] == "fakeLaunchConfig" and grid_x["type"] == "uint32_t"
    element = _field(schema, "fakeLaunchEx_enter", "config_extra_1_z")
    assert element["path"] == ["extra", 1, "z"]


def test_unions_are_bytes_with_no_machine_path(files, schema):
    union = _field(schema, "fakeBits_enter", "value_union")
    assert union["encoding"] == "UInt8FixedArray" and union["count"] == 8
    assert union["type"] == "fakeValue::(unnamed union)" and "path" not in union
    assert " at " not in files["fake_probes.json"]


def test_strings_and_capture_levels(schema):
    path = _field(schema, "fakeLoad_enter", "path")
    assert path["encoding"] == "CountedString" and path["capped"]
    assert path["capture"] == "full"
    kname = _field(schema, "fakeGetFunction_enter", "kname")
    assert kname["encoding"] == "String" and "capped" not in kname
    stream = _field(schema, "fakeLoad_enter", "stream")
    assert stream["encoding"] == "Pointer" and stream["capture"] == "scalars"


def test_outputs_results_and_exception(schema):
    name = _field(schema, "fakeName_exit", "name")
    assert name["role"] == "output" and name["on_success"] and name["capture"] == "full"
    result = _field(schema, "fakeName_exit", "result")
    assert result["capture"] == "none" and "on_success" not in result
    exception = _field(schema, "fakeName_exit", "exception")
    assert exception["role"] == "exception" and exception["type"] == "bool"


def test_reserved_names_keep_their_parameter(schema):
    renamed = _field(schema, "fakeReserved_enter", "arg_timestamp")
    assert renamed["param"] == "timestamp" and renamed["role"] == "argument"
