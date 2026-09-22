"""Tests for perfxpert.runtime.recursion_guard (spec §5.8 / review N8)."""

import os

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
    assert recursion_guard.in_opencode_session() is True
    assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") != "1"


def test_context_manager_cleans_up(monkeypatch):
    original_state = recursion_guard.in_opencode_session()
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    with recursion_guard.opencode_session():
        assert recursion_guard.in_opencode_session() is True
        assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") != "1"
    assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") != "1"
    assert recursion_guard.in_opencode_session() is original_state


def test_mark_entry_does_not_mutate_process_environment(monkeypatch):
    original = os.environ.get("PERFXPERT_IN_OPENCODE_SESSION")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    recursion_guard.mark_entry()
    assert os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") == original
    assert recursion_guard.in_opencode_session() is True


def test_clear_only_resets_local_override(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    recursion_guard.mark_entry()
    recursion_guard.clear()
    assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "1"
    assert recursion_guard.in_opencode_session() is True


def test_context_manager_restores_local_state_without_touching_environment(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "0")
    assert recursion_guard.in_opencode_session() is False

    with recursion_guard.opencode_session():
        assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "0"
        assert recursion_guard.in_opencode_session() is True

    assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "0"
    assert recursion_guard.in_opencode_session() is False


def test_context_manager_preserves_inherited_true_environment(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "1")
    with recursion_guard.opencode_session():
        assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "1"
        assert recursion_guard.in_opencode_session() is True
    assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "1"
    assert recursion_guard.in_opencode_session() is True


def test_subprocess_env_sets_inherited_marker(monkeypatch):
    monkeypatch.setenv("PERFXPERT_IN_OPENCODE_SESSION", "0")
    env = recursion_guard.subprocess_env(os.environ)
    assert env["PERFXPERT_IN_OPENCODE_SESSION"] == "1"
    assert os.environ["PERFXPERT_IN_OPENCODE_SESSION"] == "0"
