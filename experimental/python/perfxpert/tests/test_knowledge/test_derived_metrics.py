"""Tests for knowledge/derived_metrics.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "derived_metrics.schema.json"
)


def test_derived_metrics_loads():
    entries = load_yaml("derived_metrics")
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_derived_metrics_validates():
    entries = load_yaml("derived_metrics")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entries, schema)


def test_derived_metrics_covers_fetch_and_write_size():
    entries = load_yaml("derived_metrics")
    names = {e["name"] for e in entries}
    assert "FETCH_SIZE" in names
    assert "WRITE_SIZE" in names


def test_components_reference_existing_counters():
    """Every component must exist in counter_catalog.yaml."""
    entries = load_yaml("derived_metrics")
    catalog = load_yaml("counter_catalog")
    known_counters = {c["name"] for c in catalog}
    for m in entries:
        for c in m["components"]:
            assert c in known_counters, (
                f"derived_metrics[{m['name']}] references unknown counter: {c}"
            )
