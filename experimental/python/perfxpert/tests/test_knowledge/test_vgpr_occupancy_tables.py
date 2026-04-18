"""Tests for knowledge/vgpr_occupancy_tables.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "vgpr_occupancy_tables.schema.json"
)


def test_vgpr_occupancy_tables_loads_without_error():
    data = load_yaml("vgpr_occupancy_tables")
    assert data is not None


def test_vgpr_occupancy_tables_validates_against_schema():
    data = load_yaml("vgpr_occupancy_tables")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_vgpr_occupancy_tables_has_expected_content():
    data = load_yaml("vgpr_occupancy_tables")
    assert "gfx942" in data
    assert "gfx950" in data
    # CDNA3 table has 7 entries
    assert len(data["gfx942"]) == 7
