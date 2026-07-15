"""Tests for knowledge/pc_sampling_sample_types.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "pc_sampling_sample_types.schema.json"
)


def test_pc_sampling_sample_types_loads():
    entries = load_yaml("pc_sampling_sample_types")
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_pc_sampling_sample_types_validates():
    entries = load_yaml("pc_sampling_sample_types")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entries, schema)


def test_all_three_types_present():
    entries = load_yaml("pc_sampling_sample_types")
    names = {e["name"] for e in entries}
    assert names == {"ISSUED", "LATENCY", "INDETERMINATE"}


def test_latency_is_only_actionable_type():
    entries = load_yaml("pc_sampling_sample_types")
    for e in entries:
        if e["name"] == "LATENCY":
            assert e.get("actionable") is True
