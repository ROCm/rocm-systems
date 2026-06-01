"""End-of-Phase-2 integration smoke test.

Covers:
  * All 5 providers resolvable via registry + dry_run path
  * All 7 fence roles build deterministically
  * Conversation to_dict / from_dict round-trip
  * load_config() succeeds with defaults
  * Registry is populated after provider imports
  * Total runtime budget < 3 seconds (mocked/dry-run only)
"""

import time

import pytest

from perfxpert.config import PerfXpertConfig, load_config
from perfxpert.conversation import Conversation
from perfxpert.fence import FenceBuilder
from perfxpert.providers import (
    DryRunResponse,
    Provider,
    ProviderResponse,
    get_provider,
    list_providers,
)

# Trigger registration by importing each provider module
import perfxpert.providers.anthropic_provider  # noqa: F401
import perfxpert.providers.openai_provider  # noqa: F401
import perfxpert.providers.ollama_provider  # noqa: F401
import perfxpert.providers.private_provider  # noqa: F401
import perfxpert.providers.opencode_provider  # noqa: F401


_EXPECTED_PROVIDERS = {"anthropic", "openai", "ollama", "private", "opencode"}
_EXPECTED_ROLES = [
    "root",
    "analysis",
    "recommendation",
    "correctness",
    "compute_specialist",
    "memory_specialist",
    "latency_specialist",
]


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch, tmp_path):
    """Give each provider a placeholder key so instantiation succeeds."""
    monkeypatch.setenv("PERFXPERT_LLM_ANTHROPIC_KEY", "sk-smoke")
    monkeypatch.setenv("PERFXPERT_LLM_OPENAI_KEY", "sk-smoke")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_URL", "https://smoke.local/v1")
    monkeypatch.setenv("PERFXPERT_LLM_PRIVATE_MODEL", "smoke")
    monkeypatch.setenv("PERFXPERT_OPENCODE_PATH", "/tmp/opencode-smoke")
    monkeypatch.delenv("PERFXPERT_IN_OPENCODE_SESSION", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def test_registry_contains_all_five_providers():
    registered = set(list_providers().keys())
    assert _EXPECTED_PROVIDERS.issubset(registered), (
        f"missing: {_EXPECTED_PROVIDERS - registered}"
    )


def test_all_five_providers_dry_run_no_network():
    for name in _EXPECTED_PROVIDERS:
        provider = get_provider(name)
        assert isinstance(provider, Provider)
        result = provider.complete(
            [{"role": "user", "content": "hello"}],
            dry_run=True,
        )
        assert result is DryRunResponse


def test_all_seven_fence_roles_build():
    fb = FenceBuilder()
    for role in _EXPECTED_ROLES:
        text = fb.build(role)
        assert text
        assert len(text) > 100


def test_fence_determinism_end_to_end():
    fb = FenceBuilder()
    for role in _EXPECTED_ROLES:
        assert fb.build(role) == fb.build(role)


def test_conversation_round_trip():
    c = Conversation(api_key="sk-secret")
    c.add_user("ping")
    c.add_assistant("pong")
    payload = c.to_dict()
    assert "api_key" not in payload
    restored = Conversation.from_dict(payload)
    assert restored.messages == c.messages
    assert restored.turn_count == c.turn_count


def test_load_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, PerfXpertConfig)
    assert cfg.provider in _EXPECTED_PROVIDERS


def test_total_runtime_budget_under_three_seconds():
    t0 = time.perf_counter()

    fb = FenceBuilder()
    for role in _EXPECTED_ROLES:
        fb.build(role)

    for name in _EXPECTED_PROVIDERS:
        provider = get_provider(name)
        provider.complete([{"role": "user", "content": "x"}], dry_run=True)

    c = Conversation()
    for i in range(5):
        c.add_user(f"u{i}")
        c.add_assistant(f"a{i}")
    _ = c.to_dict()

    _ = load_config()

    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"phase-2 smoke exceeded 3s budget: {elapsed:.2f}s"
