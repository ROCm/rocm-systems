"""Cycle-4 B2 — LLM model id is parameterizable via env var.

Prior to this fix, `--llm anthropic` would hit a hardcoded stale
`claude-sonnet-4-20250514` string in two places (`agents/framework.py`
and `providers/anthropic_provider.py`), returning `400 model_not_found`
at runtime. The fix:

  1. Built-in defaults updated to the current-default model id.
  2. Each provider honours ``PERFXPERT_<PROVIDER>_MODEL`` (new), the
     pre-existing ``PERFXPERT_AGENTS_MODEL_<PROVIDER>`` (agents SDK
     layer), and ``PERFXPERT_LLM_MODEL`` (global).

These tests pin the contract end-to-end so the resolver is stable
across future model bumps.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every model-related env var the test suite might have set.

    Tests opt IN to individual env vars via ``monkeypatch.setenv``; the
    fixture's job is to ensure we never accidentally pick up a value
    leaked from the developer's shell (for example
    ``PERFXPERT_LLM_MODEL=gpt-3.5-turbo`` set for an unrelated test).
    """
    for key in list(os.environ):
        if key.startswith("PERFXPERT_") and key.endswith("_MODEL"):
            monkeypatch.delenv(key, raising=False)
        if key in ("PERFXPERT_LLM_MODEL",):
            monkeypatch.delenv(key, raising=False)
    yield


# ------------------------------------------------------------------
# framework._resolve_model (agents-SDK layer, hot path for --llm)
# ------------------------------------------------------------------


def test_agents_resolve_uses_builtin_default_anthropic() -> None:
    from perfxpert.agents.framework import _resolve_model

    # The regression: the built-in must NOT be the stale hardcode any more.
    assert _resolve_model("anthropic") != "claude-sonnet-4-20250514", (
        "built-in default for `anthropic` is still the stale "
        "`claude-sonnet-4-20250514` — cycle-4 blocker B2"
    )
    assert _resolve_model("anthropic") == "claude-sonnet-4-5"


def test_agents_resolve_honours_provider_specific_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfxpert.agents.framework import _resolve_model

    monkeypatch.setenv("PERFXPERT_ANTHROPIC_MODEL", "claude-opus-4-7")
    assert _resolve_model("anthropic") == "claude-opus-4-7"

    monkeypatch.setenv("PERFXPERT_OPENAI_MODEL", "gpt-4o")
    assert _resolve_model("openai") == "gpt-4o"


def test_agents_resolve_provider_specific_wins_over_llm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfxpert.agents.framework import _resolve_model

    # If both are set, the per-provider env var wins.
    monkeypatch.setenv("PERFXPERT_LLM_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("PERFXPERT_ANTHROPIC_MODEL", "claude-sonnet-4-5")
    assert _resolve_model("anthropic") == "claude-sonnet-4-5"
    # …and the global value is used for a provider without a dedicated var.
    monkeypatch.delenv("PERFXPERT_ANTHROPIC_MODEL")
    assert _resolve_model("anthropic") == "gpt-3.5-turbo"


def test_agents_resolve_agents_specific_wins_over_provider_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from perfxpert.agents.framework import _resolve_model

    # Legacy precedence: PERFXPERT_AGENTS_MODEL_<PROVIDER> is still the
    # top-priority override (it targets the agents-SDK layer specifically).
    monkeypatch.setenv("PERFXPERT_ANTHROPIC_MODEL", "claude-opus-4-5")
    monkeypatch.setenv("PERFXPERT_AGENTS_MODEL_ANTHROPIC", "claude-haiku-4-5")
    assert _resolve_model("anthropic") == "claude-haiku-4-5"


# ------------------------------------------------------------------
# providers.anthropic_provider._resolve_default_model
# ------------------------------------------------------------------


def test_anthropic_provider_default_model_not_stale() -> None:
    from perfxpert.providers import anthropic_provider as mod

    # Same regression: built-in default must not be a 2024-stale id.
    assert mod._BUILTIN_DEFAULT_MODEL != "claude-3-5-sonnet-20241022"
    assert mod._resolve_default_model() == "claude-sonnet-4-5"


def test_anthropic_provider_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfxpert.providers import anthropic_provider as mod

    monkeypatch.setenv("PERFXPERT_ANTHROPIC_MODEL", "claude-opus-4-7")
    assert mod._resolve_default_model() == "claude-opus-4-7"


def test_anthropic_provider_llm_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfxpert.providers import anthropic_provider as mod

    monkeypatch.setenv("PERFXPERT_LLM_MODEL", "claude-some-future")
    assert mod._resolve_default_model() == "claude-some-future"


# ------------------------------------------------------------------
# providers.openai_provider._resolve_default_model
# ------------------------------------------------------------------


def test_openai_provider_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfxpert.providers import openai_provider as mod

    monkeypatch.setenv("PERFXPERT_OPENAI_MODEL", "gpt-4o")
    assert mod._resolve_default_model() == "gpt-4o"


def test_openai_provider_llm_model_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from perfxpert.providers import openai_provider as mod

    monkeypatch.setenv("PERFXPERT_LLM_MODEL", "gpt-99-future")
    assert mod._resolve_default_model() == "gpt-99-future"


def test_openai_provider_builtin_default() -> None:
    from perfxpert.providers import openai_provider as mod

    assert mod._resolve_default_model() == "gpt-4o-mini"


# ------------------------------------------------------------------
# doctor prints resolved models
# ------------------------------------------------------------------


def test_doctor_resolved_models_helper_covers_all_providers() -> None:
    from perfxpert.__main__ import _resolved_models

    result = _resolved_models()
    for prov in ("anthropic", "openai", "ollama", "private", "opencode"):
        assert prov in result, f"{prov} missing from doctor's resolved-model map"
        assert result[prov], f"{prov} resolved to empty/None model id"
    assert result["anthropic"] == "claude-sonnet-4-5"
