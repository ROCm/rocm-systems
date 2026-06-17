"""Tests for knowledge/bottleneck_types.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "bottleneck_types.schema.json"
)


def test_bottleneck_types_has_required_classes():
    types = load_yaml("bottleneck_types")
    names = {entry["name"] for entry in types}
    required = {"compute", "memory_transfer", "latency", "mixed", "api_overhead"}
    missing = required - names
    assert not missing, f"missing bottleneck classes: {missing}"


def test_each_entry_has_signatures_and_priority():
    types = load_yaml("bottleneck_types")
    for entry in types:
        assert "name" in entry
        assert "signatures" in entry
        assert isinstance(entry["signatures"], list)
        assert len(entry["signatures"]) >= 1
        assert "priority_hint" in entry
        assert entry["priority_hint"] in ("high", "medium", "low")


def test_validates_against_schema():
    types = load_yaml("bottleneck_types")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(types, schema)
