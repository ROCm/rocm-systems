"""Tests for perfxpert.providers._reference_guide.

The monolithic llm-reference-guide.md is retired; the loader now raises
unconditionally. Callers must use `perfxpert.agents.fence.load_fence_slice`.
"""

import pytest

from perfxpert.providers._reference_guide import (
    load_reference_guide,
    ReferenceGuideNotFoundError,
)


def test_load_reference_guide_always_raises():
    """In Phase 7.1 the loader raises — the monolithic guide is gone."""
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()


def test_load_reference_guide_ignores_legacy_env(monkeypatch, tmp_path):
    """PERFXPERT_LEGACY / ROCINSIGHT_LLM_REFERENCE_GUIDE no longer flip any switch."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    override = tmp_path / "my-guide.md"
    override.write_text("# Custom Fence\nHello legacy.\n")
    monkeypatch.setenv("ROCINSIGHT_LLM_REFERENCE_GUIDE", str(override))
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()
