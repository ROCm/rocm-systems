"""Shared fixtures — mocked providers, stub tools, fake handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


@dataclass
class FakeProviderResponse:
    """Emulates what framework.run_agent returns under mocked providers."""
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    structured_output: Optional[Dict[str, Any]] = None
    handoff: Optional[str] = None


@pytest.fixture
def fake_provider(monkeypatch) -> MagicMock:
    """Return a MagicMock the facade will use instead of a real SDK client.

    Tests set `fake_provider.return_value = FakeProviderResponse(...)` to
    script the LLM output for a single-turn isolation test.
    """
    from perfxpert.agents import framework

    mock = MagicMock()
    monkeypatch.setattr(framework, "_sdk_invoke", mock)
    return mock


@pytest.fixture
def fake_tool_registry() -> Dict[str, Callable]:
    """A simple tool registry fixtures can mutate for allowlist tests."""
    return {}
