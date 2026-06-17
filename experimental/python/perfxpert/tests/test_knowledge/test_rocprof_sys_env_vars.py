"""Tests for knowledge/rocprof_sys_env_vars.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "rocprof_sys_env_vars.schema.json"
)


def test_rocprof_sys_env_vars_loads():
    entries = load_yaml("rocprof_sys_env_vars")
    assert isinstance(entries, list)
    assert len(entries) >= 1


def test_rocprof_sys_env_vars_validates():
    entries = load_yaml("rocprof_sys_env_vars")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entries, schema)


def test_pc_sampling_beta_var_present():
    """PC sampling requires ROCPROFILER_PC_SAMPLING_BETA_ENABLED — must be documented."""
    entries = load_yaml("rocprof_sys_env_vars")
    names = {e["name"] for e in entries}
    assert "ROCPROFILER_PC_SAMPLING_BETA_ENABLED" in names
