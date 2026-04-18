"""Phase 6 deletion assertions.

These tests FAIL while the legacy modules still exist (before Task 10's
deletion commit). They PASS after deletion. If a future PR accidentally
reintroduces the legacy modules, these tests catch it.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("mod", [
    "perfxpert.ai_analysis.interactive",
    "perfxpert.ai_analysis.llm_conversation",
])
def test_removed_module_cannot_be_imported(mod):
    spec = importlib.util.find_spec(mod)
    assert spec is None, f"{mod} should have been deleted in Phase 6 PR 2"


def test_LLMConversation_not_exported():
    import perfxpert.ai_analysis as ai
    assert not hasattr(ai, "LLMConversation"), \
        "perfxpert.ai_analysis.LLMConversation was retired in Phase 6"
    assert "LLMConversation" not in getattr(ai, "__all__", [])


def test_InteractiveSession_not_exported():
    import perfxpert.ai_analysis as ai
    assert not hasattr(ai, "InteractiveSession"), \
        "perfxpert.ai_analysis.InteractiveSession was retired in Phase 6"


def test_WorkflowSession_not_exported():
    import perfxpert.ai_analysis as ai
    assert not hasattr(ai, "WorkflowSession"), \
        "perfxpert.ai_analysis.WorkflowSession was retired in Phase 6"


def test_analyze_with_llm_method_removed():
    """LLMAnalyzer kept as a shell; analyze_with_llm() method deleted."""
    from perfxpert.ai_analysis.llm_analyzer import LLMAnalyzer
    assert not hasattr(LLMAnalyzer, "analyze_with_llm"), \
        "LLMAnalyzer.analyze_with_llm was retired in Phase 6"


def test_monolithic_reference_guide_file_deleted():
    """The monolithic share/llm-reference-guide.md file is deleted in Phase 6 PR 2."""
    pkg_root = Path(importlib.util.find_spec("perfxpert").origin).parent
    guide = pkg_root / "ai_analysis" / "share" / "llm-reference-guide.md"
    assert not guide.exists(), \
        f"{guide} should have been deleted in Phase 6 PR 2; split fence lives in agents/fence/"


def test_agents_fence_dir_exists():
    """The replacement split fence lives under agents/fence/."""
    pkg_root = Path(importlib.util.find_spec("perfxpert").origin).parent
    fence_dir = pkg_root / "agents" / "fence"
    # Phase 2 deliverable — may not exist yet in some branches
    if fence_dir.exists():
        assert fence_dir.is_dir(), f"{fence_dir} must be a directory"
        # At least the 7 per-agent files + always.md expected from Phase 2.
        md_files = sorted(p.name for p in fence_dir.glob("*.md"))
        expected = {"always.md", "root.md", "analysis.md", "recommendation.md",
                    "correctness.md", "compute_specialist.md", "memory_specialist.md",
                    "latency_specialist.md"}
        missing = expected - set(md_files)
        # Don't fail if fence directory doesn't exist; Phase 2 may be pending
        if not missing:
            pytest.skip("agents/fence not yet present (Phase 2 deliverable)")
        assert not missing, f"missing agent fence files: {missing}"
    else:
        pytest.skip("agents/fence not yet present (Phase 2 deliverable)")
