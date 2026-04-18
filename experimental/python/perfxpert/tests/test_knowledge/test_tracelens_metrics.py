"""Tests for knowledge/tracelens_metrics.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "tracelens_metrics.schema.json"
)


def test_tracelens_metrics_loads_without_error():
    data = load_yaml("tracelens_metrics")
    assert data is not None


def test_tracelens_metrics_validates_against_schema():
    data = load_yaml("tracelens_metrics")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_tracelens_metrics_has_expected_content():
    data = load_yaml("tracelens_metrics")
    assert data["thresholds"]["idle_high"] == 0.20
    assert data["thresholds"]["wasted_high"] == 0.05
