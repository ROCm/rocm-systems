"""Tests for knowledge/proven_optimizations.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml


SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "proven_optimizations.schema.json"
)


def test_proven_optimizations_loads_without_error():
    data = load_yaml("proven_optimizations")
    assert data is not None


def test_proven_optimizations_validates_against_schema():
    data = load_yaml("proven_optimizations")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_proven_optimizations_has_ten_seed_cases():
    data = load_yaml("proven_optimizations")
    ids = [entry["id"] for entry in data]
    assert len(ids) == 10
    assert len(set(ids)) == len(ids)
    assert "mfma_enablement" in ids
    assert "hip_stream_overlap" in ids
