"""Tests for knowledge/pmc_limits.yaml."""

import json
from pathlib import Path

import jsonschema

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "pmc_limits.schema.json"
)

_COMMON_MI_LIMITS = {
    "SQ": 8,
    "TA": 2,
    "TD": 2,
    "TCP": 4,
    "TCC": 4,
    "CPC": 2,
    "CPF": 2,
    "SPI": 6,
    "GRBM": 2,
    "GDS": 4,
}

_EXPECTED_GPU_ARCH_LIMITS = {
    "gfx908": _COMMON_MI_LIMITS,
    "gfx90a": _COMMON_MI_LIMITS,
    "gfx940": _COMMON_MI_LIMITS,
    "gfx941": _COMMON_MI_LIMITS,
    "gfx942": _COMMON_MI_LIMITS,
    "gfx950": _COMMON_MI_LIMITS,
    "gfx1151": {
        "SQ": 8,
        "TA": 2,
        "GRBM": 2,
        "CPC": 2,
        "GCEA": 2,
        "TCP": 4,
        "SPI": 6,
        "GL1A": 4,
        "GL1C": 4,
        "GL2A": 4,
        "GL2C": 4,
    },
}


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
    required = set().union(*[set(v) for v in _EXPECTED_GPU_ARCH_LIMITS.values()])
    missing = required - blocks
    assert not missing, f"missing blocks in pmc_limits: {missing}"


def test_gpu_arch_limits_match_rocprofiler_compute_source_shape():
    data = load_yaml("pmc_limits")
    assert data["gpu_arch_limits"] == _EXPECTED_GPU_ARCH_LIMITS


def test_per_block_fallback_limits_cover_source_blocks():
    data = load_yaml("pmc_limits")
    limits = data["per_block_limits"]
    expected = set().union(*[set(v) for v in _EXPECTED_GPU_ARCH_LIMITS.values()])

    for block in expected:
        assert limits[block]["limit"] >= 1


def test_gfx950_tcc_channels_is_not_a_counter_block_limit():
    data = load_yaml("pmc_limits")
    assert "TCC_channels" not in data["gpu_arch_limits"]["gfx950"]
    assert "TCC_channels" not in data["per_block_limits"]
    assert data["gpu_arch_limits"]["gfx950"]["TCC"] == 4
