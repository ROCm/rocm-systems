"""Tests for knowledge/sol_metrics.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "sol_metrics.schema.json"
)


def test_sol_metrics_loads_without_error():
    data = load_yaml("sol_metrics")
    assert data is not None


def test_sol_metrics_validates_against_schema():
    data = load_yaml("sol_metrics")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_sol_metrics_has_expected_content():
    data = load_yaml("sol_metrics")
    names = {m["name"] for m in data}
    assert "VALU_UTIL" in names
    assert "MFMA_UTIL" in names
    assert len(data) >= 5
