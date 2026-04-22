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
    """The loader now raises unconditionally — the monolithic guide is gone."""
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()


def test_load_reference_guide_ignores_legacy_env(monkeypatch, tmp_path):
    """Regression guard: removed legacy env vars must not flip any switch."""
    # Regression guard — assert removed legacy env vars stay inert.
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")  # regression guard
    override = tmp_path / "my-guide.md"
    override.write_text("# Custom Fence\nsentinel content.\n")
    # The pre-rename reference-guide env var (fully removed)
    # must never revive the deleted monolithic loader.
    monkeypatch.setenv("PERFXPERT_LEGACY_REFERENCE_GUIDE", str(override))  # regression guard
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()
