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
    assert "gpu_arch_limits" in data
    assert len(data["gpu_arch_limits"]) >= 1
    assert "per_block_limits" in data
    assert len(data["per_block_limits"]) >= 1


def test_pmc_limits_validates_against_schema():
    data = load_yaml("pmc_limits")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)


def test_pmc_limits_covers_all_blocks():
    data = load_yaml("pmc_limits")
    blocks = set(data["per_block_limits"].keys())
    required = set().union(*[set(v) for v in _reference_perfmon_config().values()])
    missing = required - blocks
    assert not missing, f"missing blocks in pmc_limits: {missing}"


def test_gpu_arch_limits_match_rocprofiler_compute_mi_gpu_spec():
    data = load_yaml("pmc_limits")
    gpu_arch_limits = data["gpu_arch_limits"]

    for gfx_id, perfmon in _reference_perfmon_config().items():
        assert gfx_id in gpu_arch_limits
        for block, expected_limit in perfmon.items():
            assert gpu_arch_limits[gfx_id][block] == expected_limit


def test_per_block_fallback_limits_cover_source_blocks():
    data = load_yaml("pmc_limits")
    limits = data["per_block_limits"]
    expected = set().union(*[set(v) for v in _reference_perfmon_config().values()])

    for block in expected:
        assert limits[block]["limit"] >= 1


def test_per_block_fallback_limits_match_minimum_arch_limit():
    data = load_yaml("pmc_limits")
    limits_by_block = {}
    for perfmon in data["gpu_arch_limits"].values():
        for block, limit in perfmon.items():
            limits_by_block.setdefault(block, []).append(limit)

    for block, arch_limits in limits_by_block.items():
        assert data["per_block_limits"][block]["limit"] == min(arch_limits), block


def test_channel_metadata_is_not_a_counter_block_limit():
    data = load_yaml("pmc_limits")
    for arch_limits in data["gpu_arch_limits"].values():
        assert not any(block.endswith("_channels") for block in arch_limits)
    assert not any(block.endswith("_channels") for block in data["per_block_limits"])
