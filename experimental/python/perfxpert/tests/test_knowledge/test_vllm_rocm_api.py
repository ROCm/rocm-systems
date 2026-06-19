"""Tests for knowledge/vllm_rocm_api.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "vllm_rocm_api.schema.json"
)


def test_vllm_rocm_api_loads_without_error():
    data = load_yaml("vllm_rocm_api")
    assert data is not None


def test_vllm_rocm_api_validates_against_schema():
    data = load_yaml("vllm_rocm_api")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_vllm_rocm_api_has_expected_content():
    data = load_yaml("vllm_rocm_api")
    pitfall_ids = {p["id"] for p in data["pitfalls"]}
    assert "pin_memory_not_param" in pitfall_ids
