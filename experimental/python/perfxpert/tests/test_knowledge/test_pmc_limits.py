"""Tests for knowledge/pmc_limits.yaml."""

import json
from pathlib import Path

import jsonschema
import yaml

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = Path(__file__).parent.parent.parent / "perfxpert" / "knowledge" / "_schemas" / "pmc_limits.schema.json"


def _mi_gpu_spec_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "projects" / "rocprofiler-compute" / "src" / "utils" / "mi_gpu_spec.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("projects/rocprofiler-compute/src/utils/mi_gpu_spec.yaml")


def _reference_perfmon_config():
    data = yaml.safe_load(_mi_gpu_spec_path().read_text())
    configs = {}
    for series in data["mi_gpu_spec"]:
        for arch in series.get("gpu_archs", []):
            perfmon = arch.get("perfmon_config") or {}
            configs[arch["gpu_arch"]] = {
                block: limit
                for block, limit in perfmon.items()
                if isinstance(limit, int) and not block.endswith("_channels")
            }
    return configs


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


def test_default_limits_are_conservative_positive_values():
    data = load_yaml("pmc_limits")
    for block, info in data["per_block_limits"].items():
        assert info["limit"] > 0, block


def test_default_limits_match_minimum_arch_limit():
    data = load_yaml("pmc_limits")
    for block, info in data["per_block_limits"].items():
        arch_limits = info.get("arch_limits", {})
        if arch_limits:
            assert info["limit"] == min(arch_limits.values()), block


def test_arch_limits_match_rocprofiler_compute_mi_gpu_spec():
    data = load_yaml("pmc_limits")
    per_block = data["per_block_limits"]

    for gfx_id, perfmon in _reference_perfmon_config().items():
        for block, expected_limit in perfmon.items():
            assert block in per_block
            assert per_block[block]["arch_limits"][gfx_id] == expected_limit
