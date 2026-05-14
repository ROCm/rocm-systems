"""Tests for knowledge/att_output_format.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "att_output_format.schema.json"
)


def test_att_output_format_loads_without_error():
    data = load_yaml("att_output_format")
    assert data is not None


def test_att_output_format_validates_against_schema():
    data = load_yaml("att_output_format")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_att_output_format_has_expected_content():
    data = load_yaml("att_output_format")
    assert data["hitcount_minimum"] == 6400
    assert data["stall_ratio_thresholds"]["critical"] == 0.80
