"""Tests for knowledge/pc_sampling_stall_reasons.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "pc_sampling_stall_reasons.schema.json"
)


def test_pc_sampling_stall_reasons_loads_without_error():
    data = load_yaml("pc_sampling_stall_reasons")
    assert data is not None


def test_pc_sampling_stall_reasons_validates_against_schema():
    data = load_yaml("pc_sampling_stall_reasons")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_pc_sampling_stall_reasons_has_expected_content():
    data = load_yaml("pc_sampling_stall_reasons")
    codes = {entry["code"] for entry in data}
    assert "INTERLOCK_VMEM" in codes
    assert "BARRIER" in codes
    assert len(data) >= 8
