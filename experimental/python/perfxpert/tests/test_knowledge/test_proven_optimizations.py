"""Tests for knowledge/proven_optimizations.yaml."""

import json
from pathlib import Path

import jsonschema
import pytest

from perfxpert.knowledge import load_yaml

SCHEMA_PATH = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "knowledge" / "_schemas" / "proven_optimizations.schema.json"
)

EXPECTED_CASE_IDS = {
    "vgpr_reduction_compute_bound",
    "memory_coalescing_stride_fix",
    "mfma_enablement",
    "fast_math_compiler_flag",
    "lds_tiling_matmul",
    "hip_stream_overlap",
    "kernel_fusion_small_launches",
    "device_sync_removal",
    "warp_primitives_reduction",
    "cache_blocking_kernel",
}


def test_proven_optimizations_loads():
    cases = load_yaml("proven_optimizations")
    assert isinstance(cases, list)
    assert len(cases) >= 10


def test_all_10_seed_case_ids_present():
    cases = load_yaml("proven_optimizations")
    ids = {c["id"] for c in cases}
    missing = EXPECTED_CASE_IDS - ids
    assert not missing, f"missing case IDs: {missing}"


def test_validates_against_schema():
    cases = load_yaml("proven_optimizations")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(cases, schema)


def test_every_case_has_fixture_pair_declared():
    cases = load_yaml("proven_optimizations")
    for c in cases:
        fp = c["fixture_pair"]
        assert "baseline_db" in fp
        assert "optimized_db" in fp
        assert fp["baseline_db"].endswith(".db")
        assert fp["optimized_db"].endswith(".db")


def test_measured_speedup_ranges_are_sane():
    cases = load_yaml("proven_optimizations")
    for c in cases:
        lo, hi = c["measured_speedup_range"]
        assert 1.0 < lo <= hi, f"{c['id']}: bad range {lo}..{hi}"
        assert hi <= 20.0, f"{c['id']}: hi speedup {hi}× implausible"


def test_every_case_cites_source():
    cases = load_yaml("proven_optimizations")
    for c in cases:
        assert c["source_citation"], f"{c['id']}: missing source_citation"


def test_bottleneck_types_referenced_are_known():
    cases = load_yaml("proven_optimizations")
    bt = load_yaml("bottleneck_types")
    known_bt = {entry["name"] for entry in bt}
    for c in cases:
        assert c["bottleneck_type"] in known_bt, (
            f"{c['id']}: bottleneck_type {c['bottleneck_type']!r} "
            f"not in bottleneck_types.yaml: {known_bt}"
        )
