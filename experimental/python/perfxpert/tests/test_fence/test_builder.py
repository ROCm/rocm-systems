"""Tests for perfxpert.fence._builder."""

import pytest

from perfxpert.fence import FenceBuilder


VALID_ROLES = [
    "root",
    "analysis",
    "recommendation",
    "correctness",
    "compute_specialist",
    "memory_specialist",
    "latency_specialist",
]


def test_all_roles_build_nonempty():
    fb = FenceBuilder()
    for role in VALID_ROLES:
        text = fb.build(role)
        assert text, f"empty fence for role={role}"
        assert len(text) > 100, f"fence for {role} suspiciously short"


def test_always_slice_always_present():
    fb = FenceBuilder()
    for role in VALID_ROLES:
        text = fb.build(role)
        assert "# PerfXpert Always Fence" in text


def test_role_specific_section_present():
    fb = FenceBuilder()
    text = fb.build("compute_specialist")
    assert "compute" in text.lower()


def test_unknown_role_raises():
    fb = FenceBuilder()
    with pytest.raises(KeyError):
        fb.build("bogus_role")


def test_caches_identical_inputs():
    fb = FenceBuilder()
    a = fb.build("analysis")
    b = fb.build("analysis")
    assert a == b


def test_gfx_id_specialization_included():
    fb = FenceBuilder()
    text = fb.build("analysis", gfx_id="gfx942")
    assert "MI300X" in text or "gfx942" in text


def test_bottleneck_specialization_included():
    fb = FenceBuilder()
    text = fb.build("recommendation", bottleneck="memory_transfer")
    assert "memory" in text.lower()
