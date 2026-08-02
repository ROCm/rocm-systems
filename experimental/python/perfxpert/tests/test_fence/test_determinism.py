"""Determinism + size guardrail tests for FenceBuilder."""

from pathlib import Path

import pytest

from perfxpert.fence import FenceBuilder


_SLICES_DIR = (
    Path(__file__).parent.parent.parent
    / "perfxpert" / "agents" / "fence"
)


ROLES = [
    "root",
    "analysis",
    "recommendation",
    "correctness",
    "compute_specialist",
    "memory_specialist",
    "latency_specialist",
    "diff_specialist",
]


def test_bit_identical_across_three_calls():
    fb = FenceBuilder()
    for role in ROLES:
        a = fb.build(role)
        b = fb.build(role)
        c = fb.build(role)
        assert a == b == c, f"non-deterministic output for role={role}"


def test_slices_directory_is_not_empty():
    """Guard the glob below from vacuously passing if the directory moves."""
    assert sorted(_SLICES_DIR.glob("*.md")), f"no fence slices under {_SLICES_DIR}"


def test_each_slice_under_400_lines():
    for slice_path in sorted(_SLICES_DIR.glob("*.md")):
        lines = slice_path.read_text().splitlines()
        assert len(lines) <= 400, (
            f"slice {slice_path.name} has {len(lines)} lines (limit 400)"
        )


def test_composed_prose_under_400_lines():
    """The cap applies to what the agent actually receives, not per file."""
    from perfxpert.fence._builder import compose_prompt

    for role in ROLES:
        lines = compose_prompt(role).splitlines()
        assert len(lines) <= 400, (
            f"composed fence for {role} has {len(lines)} lines (limit 400)"
        )


def test_full_fence_under_60kb():
    fb = FenceBuilder()
    for role in ROLES:
        size = len(fb.build(role).encode("utf-8"))
        assert size <= 60 * 1024, f"{role} fence is {size} B (limit 60 KB)"


def test_specialization_is_stable():
    fb = FenceBuilder()
    a = fb.build("recommendation", bottleneck="memory_transfer", gfx_id="gfx942")
    b = fb.build("recommendation", bottleneck="memory_transfer", gfx_id="gfx942")
    assert a == b


def test_different_inputs_yield_different_output():
    fb = FenceBuilder()
    mi300 = fb.build("analysis", gfx_id="gfx942")
    mi350 = fb.build("analysis", gfx_id="gfx950")
    assert mi300 != mi350
