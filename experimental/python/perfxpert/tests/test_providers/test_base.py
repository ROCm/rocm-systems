"""Tests for perfxpert.providers._base — Provider ABC + ProviderResponse."""

import dataclasses

import pytest

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import DryRunResponse


def test_provider_response_is_frozen_dataclass():
    r = ProviderResponse(
        content="hello",
        provider="test",
        model="test-1",
        input_tokens=10,
        output_tokens=5,
    )
    assert dataclasses.is_dataclass(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.content = "mutated"


def test_provider_response_total_tokens_is_sum():
    r = ProviderResponse(
        content="x",
        provider="p",
        model="m",
        input_tokens=42,
        output_tokens=58,
    )
    assert r.total_tokens == 100


def test_provider_response_zero_tokens():
    r = ProviderResponse(
        content="",
        provider="p",
        model="m",
        input_tokens=0,
        output_tokens=0,
    )
    assert r.total_tokens == 0


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_subclass_must_implement_complete():
    class Partial(Provider):
        pass

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    class Good(Provider):
        def complete(self, messages, *, system="", model=None, max_tokens=None, dry_run=False):
            if dry_run:
                return DryRunResponse
            return ProviderResponse(
                content="ok",
                provider="good",
                model=model or "good-1",
                input_tokens=1,
                output_tokens=1,
            )

    g = Good()
    assert g.complete([], dry_run=True) is DryRunResponse
    r = g.complete([{"role": "user", "content": "hi"}])
    assert r.content == "ok"
    assert r.total_tokens == 2
