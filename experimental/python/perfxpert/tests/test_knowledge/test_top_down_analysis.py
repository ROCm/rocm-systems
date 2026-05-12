"""Tests for knowledge/top_down_analysis.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "top_down_analysis.schema.json"
)


def test_top_down_analysis_loads_without_error():
    data = load_yaml("top_down_analysis")
    assert data is not None


def test_top_down_analysis_validates_against_schema():
    data = load_yaml("top_down_analysis")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_top_down_analysis_has_expected_content():
    data = load_yaml("top_down_analysis")
    cat_names = {c["name"] for c in data["categories"]}
    assert "kernel_exec" in cat_names
    assert "memcpy_dominant" in data["red_flags"]
