"""Tests for perfxpert.providers._reference_guide.

The legacy monolithic guide is being deleted in Phase 6; however this loader
is kept (behind PERFXPERT_LEGACY=1) so the legacy fallback path still works
for one minor version. The loader is relocated out of llm_analyzer.py so
PR 2 can delete analyze_with_llm() cleanly.
"""

from pathlib import Path

import pytest

from perfxpert.providers._reference_guide import (
    load_reference_guide,
    ReferenceGuideNotFoundError,
)


def test_load_reference_guide_returns_empty_in_agentic_mode(monkeypatch):
    """Agentic mode: guide is split into agents/fence/*.md — this loader is legacy-only.

    When called outside legacy mode, it raises rather than silently returning
    stale content.
    """
    monkeypatch.delenv("PERFXPERT_LEGACY", raising=False)
    with pytest.raises(ReferenceGuideNotFoundError):
        load_reference_guide()


def test_load_reference_guide_missing_file_in_legacy(monkeypatch, tmp_path):
    """Even in legacy mode, if the file is absent (post-PR-2) we raise a clear error."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    monkeypatch.setenv("ROCINSIGHT_LLM_REFERENCE_GUIDE", str(tmp_path / "nope.md"))
    with pytest.raises(ReferenceGuideNotFoundError) as exc:
        load_reference_guide()
    assert "legacy" in str(exc.value).lower() or "monolith" in str(exc.value).lower()


def test_load_reference_guide_overrides_via_env(monkeypatch, tmp_path):
    """Legacy path honors ROCINSIGHT_LLM_REFERENCE_GUIDE env var."""
    monkeypatch.setenv("PERFXPERT_LEGACY", "1")
    override = tmp_path / "my-guide.md"
    override.write_text("# Custom Fence\nHello legacy.\n")
    monkeypatch.setenv("ROCINSIGHT_LLM_REFERENCE_GUIDE", str(override))
    content = load_reference_guide()
    assert "Custom Fence" in content
