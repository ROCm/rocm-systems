"""Tests for knowledge/memory_hierarchy.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "memory_hierarchy.schema.json"
)


def test_memory_hierarchy_loads_without_error():
    data = load_yaml("memory_hierarchy")
    assert data is not None


def test_memory_hierarchy_validates_against_schema():
    data = load_yaml("memory_hierarchy")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_memory_hierarchy_has_expected_content():
    data = load_yaml("memory_hierarchy")
    level_names = {l["name"] for l in data["levels"]}
    assert level_names == {"VGPR", "LDS", "L1", "L2", "HBM"}
