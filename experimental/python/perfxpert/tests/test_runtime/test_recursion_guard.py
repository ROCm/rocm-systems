"""Tests for perfxpert.runtime.recursion_guard (spec §5.8 / review N8)."""

import pytest

from perfxpert.runtime import recursion_guard


def test_fresh_environment_allows_opencode(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    # Must not raise
    recursion_guard.ensure_not_recursive("opencode")


def test_opencode_inside_opencode_is_blocked(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    with pytest.raises(recursion_guard.RecursionGuardViolation):
        recursion_guard.ensure_not_recursive("opencode")


def test_other_provider_inside_opencode_is_fine(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    # Different provider inside an opencode session is OK
    recursion_guard.ensure_not_recursive("anthropic")
    recursion_guard.ensure_not_recursive("openai")
    recursion_guard.ensure_not_recursive("ollama")


def test_mark_entry_sets_env(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    recursion_guard.mark_entry()
    import os
    assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "1"


def test_context_manager_cleans_up(monkeypatch):
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    with recursion_guard.opencode_session():
        import os
        assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") == "1"
    import os
    assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") != "1"
