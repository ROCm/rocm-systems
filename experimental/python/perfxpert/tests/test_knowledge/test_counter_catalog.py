"""Tests for knowledge/counter_catalog.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "counter_catalog.schema.json"
)


def test_counter_catalog_loads():
    entries = load_yaml("counter_catalog")
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_counter_catalog_validates_against_schema():
    entries = load_yaml("counter_catalog")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entries, schema)


def test_counter_catalog_covers_required_blocks():
    entries = load_yaml("counter_catalog")
    blocks = {entry["block"] for entry in entries}
    required = {"GRBM", "SQ", "TCP", "TCC"}
    missing = required - blocks
    assert not missing, f"missing HW blocks: {missing}"


def test_counter_catalog_has_sq_waves():
    """SQ_WAVES is the canonical occupancy counter — must be present."""
    entries = load_yaml("counter_catalog")
    names = {entry["name"] for entry in entries}
    assert "SQ_WAVES" in names
