"""Aggregate: every YAML file validates against its schema at import time.

Catches drift between YAML content and JSON schema before agents ever run.
"""

import json
from pathlib import Path

import jsonschema
import pytest
import yaml


KNOWLEDGE_DIR = (
    Path(__file__).parent.parent.parent / "perfxpert" / "knowledge"
)
SCHEMAS_DIR = KNOWLEDGE_DIR / "_schemas"

# The 21 YAML files expected per spec Appendix B
_EXPECTED_YAMLS = {
    "gpu_specs",
    "bottleneck_types",
    "amdahl_thresholds",
    "counter_catalog",
    "pmc_limits",
    "rocprofv3_counter_limits",
    "rocprof_sys_env_vars",
    "derived_metrics",
    "metric_thresholds",
    "pc_sampling_sample_types",
    "pc_sampling_pipelines",
    "pc_sampling_stall_reasons",
    "att_output_format",
    "memory_hierarchy",
    "sol_metrics",
    "top_down_analysis",
    "optimization_techniques",
    "compiler_flags",
    "vllm_rocm_api",
    "tracelens_metrics",
    "vgpr_occupancy_tables",
}


def _all_yaml_files():
    return sorted(KNOWLEDGE_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_path", _all_yaml_files(), ids=lambda p: p.name)
def test_every_yaml_validates_against_its_schema(yaml_path):
    schema_path = SCHEMAS_DIR / f"{yaml_path.stem}.schema.json"
    assert schema_path.exists(), f"No schema for {yaml_path.name}"

    data = yaml.safe_load(yaml_path.read_text())
    schema = json.loads(schema_path.read_text())
    jsonschema.validate(data, schema)


def test_all_21_expected_yaml_files_present():
    present = {p.stem for p in _all_yaml_files()}
    missing = _EXPECTED_YAMLS - present
    assert not missing, f"missing YAMLs: {missing}"


def test_every_yaml_has_a_schema():
    for yaml_path in _all_yaml_files():
        schema_path = SCHEMAS_DIR / f"{yaml_path.stem}.schema.json"
        assert schema_path.exists(), f"No schema for {yaml_path.name}"


def test_no_orphan_schemas():
    """Every schema must have a matching YAML."""
    for schema_path in SCHEMAS_DIR.glob("*.schema.json"):
        yaml_path = KNOWLEDGE_DIR / f"{schema_path.stem.replace('.schema', '')}.yaml"
        # schema.stem is 'foo.schema' — we want 'foo'
        stem = schema_path.name.replace(".schema.json", "")
        target = KNOWLEDGE_DIR / f"{stem}.yaml"
        assert target.exists(), f"Orphan schema: {schema_path} (no matching {target.name})"
