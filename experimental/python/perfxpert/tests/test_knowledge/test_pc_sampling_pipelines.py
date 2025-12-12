"""Tests for knowledge/pc_sampling_pipelines.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "pc_sampling_pipelines.schema.json"
)


def test_pc_sampling_pipelines_loads():
    entries = load_yaml("pc_sampling_pipelines")
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_pc_sampling_pipelines_validates():
    entries = load_yaml("pc_sampling_pipelines")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entries, schema)


def test_all_seven_pipelines_present():
    entries = load_yaml("pc_sampling_pipelines")
    names = {e["name"] for e in entries}
    required = {"VALU", "MATRIX", "SCALAR", "VMEM_TEX", "LDS", "FLAT", "MISC"}
    assert names == required
