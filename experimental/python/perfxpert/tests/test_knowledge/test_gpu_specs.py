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
    # Must have either peak_fp64 or peak_fp64_tflops; value 81.7 (within tolerance)
    fp64 = mi300x.get("peak_fp64_tflops") or mi300x.get("peak_fp64")
    assert fp64 is not None, "MI300X must declare peak_fp64"
    assert 80 <= fp64 <= 83, f"MI300X FP64 expected ~81.7 TFLOPS, got {fp64}"


def test_cdna4_has_160kb_lds():
    """CDNA4 (gfx950) doubled LDS to 160 KB/CU per CLAUDE.md."""
    specs = load_yaml("gpu_specs")
    mi350 = specs["gfx950"]
    lds_kb = mi350.get("lds_kb") or mi350.get("lds_per_cu_kb")
    assert lds_kb == 160, f"CDNA4 LDS expected 160 KB/CU, got {lds_kb}"
