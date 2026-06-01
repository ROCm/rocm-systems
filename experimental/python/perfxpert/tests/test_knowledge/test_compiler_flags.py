"""Tests for knowledge/compiler_flags.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "compiler_flags.schema.json"
)


def test_compiler_flags_loads_without_error():
    data = load_yaml("compiler_flags")
    assert data is not None


def test_compiler_flags_validates_against_schema():
    data = load_yaml("compiler_flags")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_compiler_flags_has_expected_content():
    data = load_yaml("compiler_flags")
    allowlisted = {f["flag"] for f in data if f["allowlist"]}
    assert "-O3" in allowlisted
    assert "-ffast-math" in allowlisted
    # Security: linker flags must be denylisted
    denylisted = {f["flag"] for f in data if not f["allowlist"]}
    assert "-Xlinker" in denylisted
