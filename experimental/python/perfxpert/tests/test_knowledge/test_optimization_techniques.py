"""Tests for knowledge/optimization_techniques.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "optimization_techniques.schema.json"
)


def test_optimization_techniques_loads_without_error():
    data = load_yaml("optimization_techniques")
    assert data is not None


def test_optimization_techniques_validates_against_schema():
    data = load_yaml("optimization_techniques")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_optimization_techniques_has_expected_content():
    data = load_yaml("optimization_techniques")
    ids = {t["id"] for t in data}
    assert "reduce_vgpr" in ids
    assert "mfma_enablement" in ids
    assert len(data) >= 7
