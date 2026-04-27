"""Tests for knowledge/rocprofv3_counter_limits.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "rocprofv3_counter_limits.schema.json"
)


def test_rocprofv3_counter_limits_loads():
    data = load_yaml("rocprofv3_counter_limits")
    assert isinstance(data, dict)
    assert "isolation_rules" in data
    assert len(data["isolation_rules"]) >= 1


def test_rocprofv3_counter_limits_validates():
    data = load_yaml("rocprofv3_counter_limits")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_tcc_derived_isolation_rule_exists():
    """FETCH_SIZE/WRITE_SIZE must be flagged as requiring isolation."""
    data = load_yaml("rocprofv3_counter_limits")
    flagged_counters = set()
    for rule in data["isolation_rules"]:
        flagged_counters.update(rule.get("counters", []))
    assert "FETCH_SIZE" in flagged_counters
    assert "WRITE_SIZE" in flagged_counters


def test_error_code_38_documented():
    data = load_yaml("rocprofv3_counter_limits")
    assert "38" in data["error_codes"]
    assert "message" in data["error_codes"]["38"]
