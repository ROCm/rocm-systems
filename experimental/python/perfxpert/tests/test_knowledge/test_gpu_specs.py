"""Tests for knowledge/gpu_specs.yaml."""

import json
from pathlib import Path

import jsonschema
import pytest

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "gpu_specs.schema.json"
)


def test_gpu_specs_loads_without_error():
    specs = load_yaml("gpu_specs")
    assert isinstance(specs, dict)
    assert len(specs) > 0


def test_gpu_specs_covers_required_archs():
    specs = load_yaml("gpu_specs")
    required_archs = {"gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100"}
    missing = required_archs - specs.keys()
    assert not missing, f"missing archs: {missing}"


def test_gpu_specs_validates_against_schema():
    specs = load_yaml("gpu_specs")
    schema = json.loads(SCHEMA_PATH.read_text())
    # Each arch entry must satisfy the per-arch schema
    for arch_id, arch_data in specs.items():
        jsonschema.validate(arch_data, schema["properties"]["arch_entry"])


def test_mi300x_fp64_peak_is_corrected():
    """CLAUDE.md correction: MI300X FP64 = 81.7 TFLOPS (NOT 163.4 which is FP32)."""
    specs = load_yaml("gpu_specs")
    mi300x = specs["gfx942"]
    fp64 = mi300x["peak_fp64_tflops"]
    assert 80 <= fp64 <= 83, f"MI300X FP64 expected ~81.7 TFLOPS, got {fp64}"


def test_cdna4_has_160kb_lds():
    """CDNA4 (gfx950) doubled LDS to 160 KB/CU per CLAUDE.md."""
    specs = load_yaml("gpu_specs")
    mi350 = specs["gfx950"]
    assert mi350["lds_kb"] == 160, f"CDNA4 LDS expected 160 KB/CU, got {mi350['lds_kb']}"


def test_default_ridge_matches_fp32_peak_over_bandwidth():
    specs = load_yaml("gpu_specs")
    for gfx_id, spec in specs.items():
        expected = round(spec["peak_fp32_tflops"] / spec["memory_bandwidth_tbs"], 1)
        assert spec["ridge_point"] == pytest.approx(expected, rel=0.01), (
            f"{gfx_id} ridge_point drifted from fp32/bandwidth ratio: "
            f"stored={spec['ridge_point']} expected={expected}"
        )


def test_runtime_occupancy_caps_exist_for_every_arch():
    specs = load_yaml("gpu_specs")
    for gfx_id, spec in specs.items():
        assert spec["lds_per_cu_kb"] > 0, gfx_id
        assert spec["vgprs_per_simd"] >= spec["max_vgprs_per_thread"], gfx_id
        assert spec["simds_per_cu"] >= 1, gfx_id
        assert spec["max_waves_per_simd"] >= 1, gfx_id


def test_schema_rejects_invalid_arch_id_pattern():
    """Schema's patternProperties should reject non-gfx identifiers."""
    schema = json.loads(SCHEMA_PATH.read_text())
    invalid_data = {
        "invalid_key_not_gfx": {
            "name": "X", "codename": "Y",
            "peak_fp64_tflops": 1, "peak_fp32_tflops": 1,
            "memory_bandwidth_tbs": 1, "cu_count": 1, "lds_kb": 64,
            "wave_size": 64, "max_vgprs_per_thread": 256, "ridge_point": 1,
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_data, schema)


def test_schema_rejects_missing_required_field():
    """Schema enforces required fields like peak_fp64_tflops."""
    schema = json.loads(SCHEMA_PATH.read_text())
    incomplete_arch = {
        "name": "fake", "codename": "X",
        # missing peak_fp64_tflops
        "peak_fp32_tflops": 1, "memory_bandwidth_tbs": 1, "cu_count": 1,
        "lds_kb": 64, "wave_size": 64, "max_vgprs_per_thread": 256, "ridge_point": 1,
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(incomplete_arch, schema["properties"]["arch_entry"])
