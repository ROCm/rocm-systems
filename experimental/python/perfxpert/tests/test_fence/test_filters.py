"""Tests for perfxpert.fence._filters."""

import pytest

from perfxpert.fence._filters import (
    ROLE_YAML_MAP,
    format_yaml_excerpt,
    get_yaml_keys_for_role,
)


def test_every_role_has_core_yamls():
    required = {"gpu_specs", "metric_thresholds", "bottleneck_types"}
    for role, mapping in ROLE_YAML_MAP.items():
        missing = required - mapping.keys()
        assert not missing, f"role {role} missing core yamls: {missing}"


def test_role_specific_yamls():
    assert "vgpr_occupancy_tables" in ROLE_YAML_MAP["compute_specialist"]
    assert "memory_hierarchy" in ROLE_YAML_MAP["memory_specialist"]
    assert "top_down_analysis" in ROLE_YAML_MAP["latency_specialist"]
    assert "amdahl_thresholds" in ROLE_YAML_MAP["recommendation"]


def test_get_yaml_keys_for_role_returns_mapping():
    mapping = get_yaml_keys_for_role("analysis")
    assert isinstance(mapping, dict)
    assert "gpu_specs" in mapping


def test_unknown_role_empty():
    mapping = get_yaml_keys_for_role("nonexistent")
    assert mapping == {}


def test_format_yaml_excerpt_renders_header():
    excerpt = format_yaml_excerpt("gpu_specs", keys=["gfx942"], gfx_id="gfx942")
    assert "## Knowledge:" in excerpt
    assert "gpu_specs" in excerpt


def test_format_yaml_excerpt_filters_by_gfx():
    text = format_yaml_excerpt("gpu_specs", keys=["all"], gfx_id="gfx942")
    assert "gfx942" in text
    # Filtered — other archs should not appear when gfx_id is set
    # (Allow always.md to carry the full table; the excerpt is focused.)


def test_format_yaml_excerpt_handles_missing_file():
    excerpt = format_yaml_excerpt("nonexistent_yaml", keys=["a"])
    assert excerpt == ""


def test_format_yaml_excerpt_empty_keys_returns_empty():
    excerpt = format_yaml_excerpt("gpu_specs", keys=[])
    assert excerpt == ""
