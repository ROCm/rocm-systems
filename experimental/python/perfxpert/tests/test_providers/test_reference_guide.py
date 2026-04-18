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
    """Regression guard: both env vars removed in Phase 7.1 must not flip any switch."""
    # Regression guard — assert env vars removed in Phase 7.1 stay inert.
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")  # regression guard
    override = tmp_path / "my-guide.md"
    override.write_text("# Custom Fence\nsentinel content.\n")
    monkeypatch.setenv("ROCINSIGHT_LLM_REFERENCE_GUIDE", str(override))  # regression guard
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()
