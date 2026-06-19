"""Tests for perfxpert.providers.registry."""

import pytest

from perfxpert.providers import registry
from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import DryRunResponse


class _StubProvider(Provider):
    def complete(self, messages, *, system="", model=None, max_tokens=None, dry_run=False):
        if dry_run:
            return DryRunResponse
        return ProviderResponse(
            content="stub",
            provider="stub",
            model=model or "stub-1",
            input_tokens=0,
            output_tokens=0,
        )


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Isolate tests — clear the module-level registry before/after."""
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_register_then_get_returns_instance():
    registry.register("stub", _StubProvider, "Stub for tests")
    inst = registry.get_provider("stub")
    assert isinstance(inst, _StubProvider)


def test_list_providers_includes_registered():
    registry.register("stub", _StubProvider, "Stub for tests")
    listing = registry.list_providers()
    assert "stub" in listing
    assert listing["stub"] == "Stub for tests"


def test_get_unknown_raises():
    with pytest.raises(KeyError) as exc:
        registry.get_provider("nonexistent")
    assert "nonexistent" in str(exc.value)


def test_re_register_overwrites_silently():
    registry.register("stub", _StubProvider, "first")
    registry.register("stub", _StubProvider, "second")
    assert registry.list_providers()["stub"] == "second"


def test_registered_instance_routes_dry_run():
    registry.register("stub", _StubProvider, "desc")
    inst = registry.get_provider("stub")
    assert inst.complete([], dry_run=True) is DryRunResponse


def test_get_provider_forwards_kwargs():
    captured = {}

    class _Captures(Provider):
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def complete(self, messages, *, system="", model=None, max_tokens=None, dry_run=False):
            return DryRunResponse

    registry.register("cap", _Captures, "desc")
    registry.get_provider("cap", foo="bar", timeout=30)
    assert captured == {"foo": "bar", "timeout": 30}
