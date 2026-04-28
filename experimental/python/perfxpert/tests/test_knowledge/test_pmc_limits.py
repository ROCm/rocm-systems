"""Tests for knowledge/pmc_limits.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "pmc_limits.schema.json"
)


def test_pmc_limits_loads():
    data = load_yaml("pmc_limits")
    assert isinstance(data, dict)
    assert "per_block_limits" in data
    assert len(data["per_block_limits"]) >= 1


def test_pmc_limits_validates_against_schema():
    data = load_yaml("pmc_limits")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_pmc_limits_covers_all_blocks():
    data = load_yaml("pmc_limits")
    blocks = set(data["per_block_limits"].keys())
    required = {"SQ", "GRBM", "TCC", "TCP", "TA", "TD"}
    missing = required - blocks
    assert not missing, f"missing blocks in pmc_limits: {missing}"


def test_sq_default_limit_is_4():
    data = load_yaml("pmc_limits")
    assert data["per_block_limits"]["SQ"]["limit"] == 4


def test_sq_gfx942_override_is_8():
    data = load_yaml("pmc_limits")
    assert data["per_block_limits"]["SQ"].get("gfx942_limit") == 8
